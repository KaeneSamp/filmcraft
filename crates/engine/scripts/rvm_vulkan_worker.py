#!/usr/bin/env python3
"""
FilmCraft Native AI Worker: Smart Matting (RVM Vulkan)
Executes neural silhouette matting using NCNN Vulkan compute on GPU (AMD Radeon / Vulkan API)
and encodes Apple ProRes 4444 (YUVA 4:4:4 12-bit) with true alpha transparency in a single pass.
Emits real-time progress events as JSON lines to stdout for FilmCraft UI consumption.
"""

import sys
import os
import time
import subprocess
import json
import argparse
import threading
import queue
import numpy as np

try:
    import cv2
    import ncnn
except ImportError:
    pass

def emit(data: dict):
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()

def run_worker(
    input_video: str,
    output_mov: str,
    param_path: str,
    model_path: str,
    start_time: str = None,
    duration_sec: float = None,
    smoothing: float = 0.70,
    feather: float = 2.0,
    green_suppression: bool = True,
    target_fps: float = 30.0,
    resolution: str = "480p",
):
    import cv2
    import ncnn

    # 1. Probe input video metadata
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of", "json", input_video,
    ]
    try:
        meta = json.loads(subprocess.check_output(probe_cmd).decode())
        st = meta["streams"][0]
        orig_w = int(st["width"])
        orig_h = int(st["height"])
        st_duration = float(st.get("duration", 0.0))
    except Exception as e:
        emit({"type": "error", "message": f"Falha ao inspecionar vídeo com ffprobe: {e}"})
        sys.exit(1)

    effective_duration = duration_sec if duration_sec and duration_sec > 0 else (st_duration if st_duration > 0 else 60.0)
    total_expected_frames = int(effective_duration * target_fps)

    if resolution == "480p":
        inf_w, inf_h = 854, 480
        target_size = 416
    else:
        inf_w, inf_h = orig_w, orig_h
        target_size = 512

    emit({
        "type": "started",
        "total": total_expected_frames,
        "width": orig_w,
        "height": orig_h,
        "fps": target_fps,
        "status": f"Iniciando pipeline Vulkan GPU ({orig_w}x{orig_h} @ {target_fps} FPS)"
    })

    # 2. FFmpeg Reader Pipe (CFR target_fps FPS)
    read_cmd = ["ffmpeg", "-v", "error"]
    if start_time:
        read_cmd += ["-ss", str(start_time)]
    read_cmd += ["-i", input_video]
    if duration_sec and duration_sec > 0:
        read_cmd += ["-t", str(duration_sec)]

    filters = [f"fps={target_fps}"]
    read_cmd += ["-vf", ",".join(filters)]
    read_cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, bufsize=orig_w * orig_h * 3 * 8)

    # 3. FFmpeg ProRes 4444 Writer Pipe (YUVA 4:4:4 12-bit transparent movie)
    os.makedirs(os.path.dirname(os.path.abspath(output_mov)), exist_ok=True)
    write_cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgra",
        "-s", f"{orig_w}x{orig_h}", "-r", str(target_fps),
        "-i", "-",
        "-c:v", "prores_ks", "-profile:v", "4",
        "-pix_fmt", "yuva444p12le",
        "-an", output_mov,
    ]
    writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=orig_w * orig_h * 4 * 8)

    # Queues for 4-stage pipeline
    q_read = queue.Queue(maxsize=16)
    q_gpu = queue.Queue(maxsize=16)
    q_post = queue.Queue(maxsize=16)

    stop_event = threading.Event()
    processed_count = [0]
    t_start = time.time()

    # Worker 1: Reader
    def reader_worker():
        frame_bytes = orig_w * orig_h * 3
        idx = 0
        try:
            while not stop_event.is_set():
                if duration_sec and idx >= total_expected_frames:
                    break
                raw = reader.stdout.read(frame_bytes)
                if not raw or len(raw) < frame_bytes:
                    break
                frame_orig = np.frombuffer(raw, dtype=np.uint8).reshape((orig_h, orig_w, 3))
                if resolution == "480p":
                    frame_inf = cv2.resize(frame_orig, (inf_w, inf_h), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_inf = frame_orig
                q_read.put((idx, frame_orig, frame_inf))
                idx += 1
        except Exception as e:
            emit({"type": "error", "message": f"Erro no leitor FFmpeg: {e}"})
        finally:
            q_read.put(None)
            try:
                reader.stdout.close()
                reader.terminate()
            except Exception:
                pass

    # Worker 2: NCNN Vulkan GPU Worker
    def gpu_worker():
        net = ncnn.Net()
        net.opt.use_vulkan_compute = True
        net.opt.use_fp16_packed = True
        net.opt.use_fp16_storage = True
        net.opt.use_fp16_arithmetic = True
        net.opt.num_threads = 4
        net.load_param(param_path)
        net.load_model(model_path)

        scale = target_size / inf_w
        w2 = target_size
        h2 = (int(inf_h * scale) + 15) // 16 * 16
        norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]

        r1 = ncnn.Mat(target_size // 2, h2 // 2, 16); r1.fill(0.0)
        r2 = ncnn.Mat(target_size // 4, h2 // 4, 20); r2.fill(0.0)
        r3 = ncnn.Mat(target_size // 8, h2 // 8, 40); r3.fill(0.0)
        r4 = ncnn.Mat(target_size // 16, h2 // 16, 64); r4.fill(0.0)

        last_alpha_ai = None

        try:
            while not stop_event.is_set():
                item = q_read.get()
                if item is None:
                    break
                idx, frame_orig, frame_inf = item
                frame_rgb = cv2.cvtColor(frame_inf, cv2.COLOR_BGR2RGB)

                # Inference with 2x frame skipping for 30+ FPS throughput
                if idx % 2 == 0 or last_alpha_ai is None:
                    in_pad = ncnn.Mat.from_pixels(frame_rgb, ncnn.Mat.PixelType.PIXEL_RGB, inf_w, inf_h)
                    in_small_pad = ncnn.Mat.from_pixels_resize(frame_rgb, ncnn.Mat.PixelType.PIXEL_RGB, inf_w, inf_h, w2, h2)
                    in_pad.substract_mean_normalize([], norm_vals)
                    in_small_pad.substract_mean_normalize([], norm_vals)

                    ex = net.create_extractor()
                    ex.input("in0", in_pad)
                    ex.input("in1", in_small_pad)
                    ex.input("in2", r1)
                    ex.input("in3", r2)
                    ex.input("in4", r3)
                    ex.input("in5", r4)

                    _, out_pha = ex.extract("out3")
                    ex.extract("out7", r1)
                    ex.extract("out8", r2)
                    ex.extract("out9", r3)
                    ex.extract("out10", r4)

                    raw_arr = np.array(out_pha)[0]
                    alpha_ai = 1.0 / (1.0 + np.exp(-raw_arr))
                    last_alpha_ai = alpha_ai
                else:
                    alpha_ai = last_alpha_ai

                q_gpu.put((idx, frame_orig, alpha_ai))
        except Exception as e:
            emit({"type": "error", "message": f"Erro de inferência NCNN Vulkan: {e}"})
        finally:
            q_gpu.put(None)

    # Worker 3: Post-processing (Temporal EMA + Feathering + Green Suppression)
    def postproc_worker():
        prev_alpha = None
        feather_k = int(round(feather)) * 2 + 1 if feather > 0 else 0

        try:
            while not stop_event.is_set():
                item = q_gpu.get()
                if item is None:
                    break
                idx, frame_orig, alpha_ai = item

                # 1. Temporal Smoothing (EMA)
                if prev_alpha is not None and smoothing > 0.0:
                    alpha_smooth = cv2.addWeighted(alpha_ai, 1.0 - smoothing, prev_alpha, smoothing, 0)
                else:
                    alpha_smooth = alpha_ai
                prev_alpha = alpha_smooth

                if green_suppression:
                    # 2. Fast SIMD Green Detection
                    hsv = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2HSV)
                    is_green = cv2.inRange(hsv, (35, 55, 45), (90, 255, 255))
                    non_green = cv2.bitwise_not(is_green)

                    # 3. Downscaled Connected Components (Chair/Subject Protection)
                    non_green_sm = cv2.resize(non_green, (214, 120), interpolation=cv2.INTER_NEAREST)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(non_green_sm)
                    best_label = 1
                    max_area = 0
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        cy = centroids[i][1]
                        if area > max_area and cy > 120 * 0.3:
                            max_area = area
                            best_label = i

                    chair_sm = (labels == best_label).astype(np.uint8) * 255
                    chair_solid = cv2.resize(chair_sm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    chair_norm = cv2.GaussianBlur(chair_solid, (9, 9), 2.5).astype(np.float32) / 255.0

                    # 4. Soft Green Suppression
                    green_soft = cv2.GaussianBlur(is_green, (7, 7), 2.0).astype(np.float32) / 255.0

                    # 5. Matte Fusion
                    alpha_res = cv2.resize(alpha_smooth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    combined_matte = np.maximum(chair_norm, alpha_res)
                    combined_matte = np.where(chair_norm > 0.05, combined_matte, alpha_res * 0.5)
                    combined_matte = combined_matte * (1.0 - green_soft)
                else:
                    combined_matte = cv2.resize(alpha_smooth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

                # 6. Anti-Aliasing Edge Feathering
                if feather_k >= 3:
                    soft_matte = cv2.GaussianBlur(combined_matte, (feather_k, feather_k), feather / 2.0)
                else:
                    soft_matte = combined_matte

                alpha_u8 = np.clip(soft_matte * 255.0, 0, 255).astype(np.uint8)

                # Despill green spill if green suppression is on
                bgr = frame_orig.copy()
                if green_suppression:
                    b, g, r = cv2.split(bgr)
                    limit = cv2.addWeighted(r, 0.5, b, 0.5, 0)
                    excess = cv2.subtract(g, limit)
                    g_clean = cv2.subtract(g, excess)
                    bgr = cv2.merge([b, g_clean, r])

                bgra = np.dstack((bgr, alpha_u8))
                q_post.put((idx, bgra))
        except Exception as e:
            emit({"type": "error", "message": f"Erro no pós-processamento: {e}"})
        finally:
            q_post.put(None)

    # Worker 4: ProRes 4444 Writer
    def writer_worker():
        try:
            while not stop_event.is_set():
                item = q_post.get()
                if item is None:
                    break
                idx, bgra = item
                writer.stdin.write(bgra.tobytes())
                processed_count[0] += 1
                cur = processed_count[0]

                if cur % 15 == 0 or cur == total_expected_frames:
                    el = time.time() - t_start
                    fps = cur / el if el > 0 else 0.0
                    eta = (total_expected_frames - cur) / fps if fps > 0 else 0.0
                    emit({
                        "type": "progress",
                        "done": cur,
                        "total": total_expected_frames,
                        "fps": round(fps, 1),
                        "eta_seconds": round(eta, 1),
                        "status": f"Recortando com Vulkan: {cur}/{total_expected_frames} ({fps:.1f} FPS)",
                    })
        except Exception as e:
            emit({"type": "error", "message": f"Erro de gravação ProRes: {e}"})
        finally:
            try:
                writer.stdin.close()
                writer.wait(timeout=5)
            except Exception:
                pass

    t_r = threading.Thread(target=reader_worker, daemon=True)
    t_g = threading.Thread(target=gpu_worker, daemon=True)
    t_p = threading.Thread(target=postproc_worker, daemon=True)
    t_w = threading.Thread(target=writer_worker, daemon=True)

    t_r.start()
    t_g.start()
    t_p.start()
    t_w.start()

    t_r.join()
    t_g.join()
    t_p.join()
    t_w.join()

    total_el = time.time() - t_start
    final_frames = processed_count[0]
    final_fps = final_frames / total_el if total_el > 0 else 0.0

    emit({
        "type": "completed",
        "output_path": output_mov,
        "frames": final_frames,
        "seconds": round(total_el, 2),
        "fps": round(final_fps, 1),
        "finished": True,
        "status": f"Concluído: {final_frames} quadros em {total_el:.1f}s ({final_fps:.1f} FPS)"
    })

def main():
    parser = argparse.ArgumentParser(description="FilmCraft AI Matting Worker")
    parser.add_argument("-i", "--input", required=True, help="Input video")
    parser.add_argument("-o", "--output", required=True, help="Output ProRes 4444 .mov")
    parser.add_argument("--param", required=True, help="NCNN .param file")
    parser.add_argument("--model", required=True, help="NCNN .bin file")
    parser.add_argument("-ss", "--start", default=None, help="Start time offset")
    parser.add_argument("-t", "--duration", type=float, default=None, help="Duration in seconds")
    parser.add_argument("--smoothing", type=float, default=0.70, help="Temporal EMA factor (0..1)")
    parser.add_argument("--feather", type=float, default=2.0, help="Feathering blur radius (px)")
    parser.add_argument("--no-green-suppression", action="store_true", help="Disable green suppression")
    parser.add_argument("--fps", type=float, default=30.0, help="Target CFR frame rate")
    parser.add_argument("--res", default="480p", choices=["480p", "720p"], help="Inference resolution")

    args = parser.parse_args()

    run_worker(
        input_video=args.input,
        output_mov=args.output,
        param_path=args.param,
        model_path=args.model,
        start_time=args.start,
        duration_sec=args.duration,
        smoothing=args.smoothing,
        feather=args.feather,
        green_suppression=not args.no_green_suppression,
        target_fps=args.fps,
        resolution=args.res,
    )

if __name__ == "__main__":
    main()
