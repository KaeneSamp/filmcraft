//! FilmCraft AI Hub & AI Execution Engine.
//!
//! Provides the core commands and background job execution infrastructure for AI tasks:
//! - `ai.tools`: lists installed AI tools and parameter schemas.
//! - `ai.run`: dispatches an AI job in the background (or synchronously when `wait: true`).
//! - `ai.status`: retrieves real-time progress metrics (done, total, ETA, FPS, status).
//! - `ai.cancel`: cleanly requests abort of a running AI job.
//! - `ai.apply`: imports generated media into the project and places transparent clips on the timeline.
//!
//! Output events stream through `filmcraft_export::Progress`, feeding both the UI panel and the
//! global jobs queue (`jobs.list`).

use std::io::BufRead;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use filmcraft_project::{ClipId, TrackKind};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, has_seq, str_p, u64_p};
use crate::{Result, Session};

/// Parameter definition for an AI tool.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiToolParam {
    pub name: String,
    pub label: String,
    pub kind: String,
    pub default: Value,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

/// Metadata and schema for an AI tool.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiToolInfo {
    pub id: String,
    pub name: String,
    pub category: String,
    pub description: String,
    pub accelerated: bool,
    pub inputs: Vec<String>,
}

/// Catalog of builtin and extensible AI tools.
pub fn available_tools() -> Vec<AiToolInfo> {
    vec![
        AiToolInfo {
            id: "matting.rvm".to_string(),
            name: "Smart Matting (RVM Vulkan)".to_string(),
            category: "Video".to_string(),
            description: "Neural video matting using RVM Vulkan with ProRes 4444 native alpha output.".to_string(),
            accelerated: true,
            inputs: vec!["clip".to_string()],
        },
        AiToolInfo {
            id: "audio.demucs".to_string(),
            name: "Vocal Isolation (Demucs)".to_string(),
            category: "Audio".to_string(),
            description: "Deep neural audio separation isolating voice from background noise.".to_string(),
            accelerated: true,
            inputs: vec!["clip".to_string()],
        },
        AiToolInfo {
            id: "caption.whisper".to_string(),
            name: "Auto-Captions (Whisper)".to_string(),
            category: "Audio".to_string(),
            description: "Local speech-to-text generating synced captions directly on the timeline.".to_string(),
            accelerated: true,
            inputs: vec!["sequence".to_string()],
        },
    ]
}

/// Locate AI worker script and NCNN Vulkan model assets on disk.
pub fn find_ai_assets() -> Option<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
    let mut search_roots = Vec::new();
    if let Some(env_d) = std::env::var_os("FILMCRAFT_AI_DIR") {
        search_roots.push(std::path::PathBuf::from(env_d));
    }
    search_roots.push(std::path::PathBuf::from("/home/kaua/Projetos/filmcraft"));
    if let Ok(cur) = std::env::current_dir() {
        search_roots.push(cur);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(p) = exe.parent() {
            search_roots.push(p.to_path_buf());
            search_roots.push(p.join(".."));
        }
    }
    if let Some(data) = crate::autosave::default_data_dir() {
        search_roots.push(data);
    }

    for base in &search_roots {
        let script = base.join("crates/engine/scripts/rvm_vulkan_worker.py");
        let param = base.join("assets/ai/models/rvm_mobilenetv3.ncnn.param");
        let bin = base.join("assets/ai/models/rvm_mobilenetv3.ncnn.bin");
        if script.exists() && param.exists() && bin.exists() {
            return Some((script, param, bin));
        }

        let script2 = base.join("scripts/rvm_vulkan_worker.py");
        let param2 = base.join("models/rvm_mobilenetv3.ncnn.param");
        let bin2 = base.join("models/rvm_mobilenetv3.ncnn.bin");
        if script2.exists() && param2.exists() && bin2.exists() {
            return Some((script2, param2, bin2));
        }
    }
    None
}

/// The command specs registered in the Engine.
pub(crate) fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.tools",
            label: "List AI Tools",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: list_tools,
            journal: false,
        },
        CommandSpec {
            id: "ai.run",
            label: "Run AI Tool",
            menu: &["Clip", "AI"],
            shortcut: None,
            params: r#"{"tool":str,"clip":id?,"wait":bool=false,"autoPlace":bool=true,"smoothing":f64=0.7,"feather":f64=2.0,"greenSuppression":bool=true,"mock":bool=false,"mockFrames":n=100}"#,
            enabled: always,
            run: run_tool,
            journal: true,
        },
        CommandSpec {
            id: "ai.status",
            label: "AI Tool Status",
            menu: &[],
            shortcut: None,
            params: r#"{"job":id}"#,
            enabled: always,
            run: tool_status,
            journal: false,
        },
        CommandSpec {
            id: "ai.cancel",
            label: "Cancel AI Tool",
            menu: &[],
            shortcut: None,
            params: r#"{"job":id}"#,
            enabled: always,
            run: cancel_tool,
            journal: true,
        },
        CommandSpec {
            id: "ai.apply",
            label: "Apply AI Result",
            menu: &[],
            shortcut: None,
            params: r#"{"path":str,"clip":id?,"track":str?}"#,
            enabled: has_seq,
            run: apply_tool,
            journal: true,
        },
        CommandSpec {
            id: "ai.preview",
            label: "Preview AI Cutout Frame",
            menu: &[],
            shortcut: None,
            params: r#"{"clip":id?,"time":f64?,"feather":f64=2.0,"greenSuppression":bool=true,"res":str="480p"}"#,
            enabled: has_seq,
            run: preview_tool,
            journal: false,
        },
    ]
}

fn list_tools(_: &mut Session, _: &Value) -> Result<Value> {
    let tools = available_tools();
    Ok(json!({ "tools": tools }))
}

/// Parse progress lines from a JSON stream (used for IPC worker communication).
pub fn parse_worker_line(line: &str, progress: &Arc<filmcraft_export::Progress>) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(line) else {
        return false;
    };

    if let Some(total) = v.get("total").and_then(Value::as_u64) {
        progress.total.store(total, Ordering::Relaxed);
    }

    if let Some(done) = v.get("done").and_then(Value::as_u64) {
        progress.done.store(done, Ordering::Relaxed);
    }

    if let Some(status) = v.get("status").and_then(Value::as_str) {
        *progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = status.to_string();
    }

    v.get("finished").and_then(Value::as_bool).unwrap_or(false)
}

fn run_tool(s: &mut Session, p: &Value) -> Result<Value> {
    let tool_id = p.get("tool").and_then(Value::as_str).unwrap_or("matting.rvm");
    let smoothing = f64_p(p, "smoothing").unwrap_or(0.70).clamp(0.0, 1.0);
    let feather = f64_p(p, "feather").unwrap_or(2.0).clamp(0.0, 8.0);
    let green_suppr = bool_p(p, "greenSuppression").unwrap_or(true);
    // Find the clip if specified or from active sequence selection
    let clip_id = p.get("clip").and_then(Value::as_u64).map(ClipId).or_else(|| {
        let q = s.active_sequence()?;
        s.state.selection.iter().copied().find(|c| {
            q.find_item(*c).is_some_and(|(t, _)| q.track(t).is_some_and(|trk| trk.kind == TrackKind::Video))
        })
    });

    let mock = bool_p(p, "mock").unwrap_or(false) || (clip_id.is_none() && str_p(p, "input").is_none());
    let mock_frames = u64_p(p, "mockFrames").unwrap_or(60).max(1);
    let wait = bool_p(p, "wait").unwrap_or(false);
    let auto_place = bool_p(p, "autoPlace").unwrap_or(true);
    let fps = f64_p(p, "fps").unwrap_or(30.0);
    let res = str_p(p, "res").unwrap_or("480p").to_string();

    let (input_path, source_in_sec, duration_sec, output_path, detected_frames, is_mock) = if let Some(inp) = str_p(p, "input") {
        let dur = f64_p(p, "duration").unwrap_or(2.0);
        let start = f64_p(p, "start").unwrap_or(0.0);
        let out = str_p(p, "output").map(str::to_string).unwrap_or_else(|| {
            let pb = std::path::PathBuf::from(inp);
            let stem = pb.file_stem().and_then(|s| s.to_str()).unwrap_or("matte");
            let dir = pb.parent().unwrap_or(std::path::Path::new("."));
            dir.join(format!("{stem}_ai_matte.mov")).to_string_lossy().to_string()
        });
        (inp.to_string(), start, dur, out, (dur * fps) as u64, mock)
    } else if let Some(cid) = clip_id {
        let q = s.active_sequence().ok_or_else(|| bad("ai.run", "no active sequence"))?;
        let (_, it) = q.find_item(cid).ok_or_else(|| bad("ai.run", "selected clip not found in sequence"))?;
        let (_, media_clip, _) = s.project.resolve_media(it.item).ok_or_else(|| bad("ai.run", "cannot resolve media for clip"))?;
        match &media_clip.media {
            filmcraft_project::MediaRef::File { path } if !mock => {
                let start_sec = it.source_in.seconds();
                let dur_sec = it.duration.seconds();
                let out = str_p(p, "output").map(str::to_string).unwrap_or_else(|| {
                    let pb = std::path::PathBuf::from(path);
                    let stem = pb.file_stem().and_then(|s| s.to_str()).unwrap_or("clip");
                    let dir = pb.parent().unwrap_or(std::path::Path::new("."));
                    dir.join(format!("{stem}_ai_matte.mov")).to_string_lossy().to_string()
                });
                (path.clone(), start_sec, dur_sec, out, (dur_sec * fps) as u64, false)
            }
            _ => {
                // If it's a generator (demo footage) or mock requested, fall back to simulation
                ("mock_demo.mp4".to_string(), 0.0, 2.0, "mock_output.mov".to_string(), mock_frames, true)
            }
        }
    } else if mock {
        ("mock_webcam.mp4".to_string(), 0.0, 2.0, "mock_output.mov".to_string(), mock_frames, true)
    } else {
        return Err(bad("ai.run", "select a video clip on timeline or specify `input`"));
    };

    let tool_info = available_tools()
        .into_iter()
        .find(|t| t.id == tool_id)
        .ok_or_else(|| bad("ai.run", format!("unknown AI tool `{tool_id}`")))?;

    // Allocate next Job ID
    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1;
    let label = format!("AI: {} (#{})", tool_info.name, id);
    let progress = Arc::new(filmcraft_export::Progress::default());
    let result = Arc::new(Mutex::new(None));

    let job = crate::Job {
        id,
        label,
        progress: progress.clone(),
        result: result.clone(),
    };

    let prog_worker = progress.clone();
    let res_worker = result.clone();
    let tool_name = tool_info.name.clone();
    let runner_out_path = output_path.clone();

    let runner = move || {
        let t0 = web_time::Instant::now();
        prog_worker.total.store(detected_frames.max(1), Ordering::Relaxed);

        if is_mock {
            let mut done = 0u64;
            let mut err = None;
            for frame in 1..=mock_frames {
                if prog_worker.cancel.load(Ordering::Relaxed) {
                    err = Some("stopped".to_string());
                    break;
                }
                if !wait {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                done = frame;
                prog_worker.done.store(done, Ordering::Relaxed);
                let elapsed = t0.elapsed().as_secs_f64().max(0.001);
                let current_fps = done as f64 / elapsed;
                *prog_worker.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!(
                    "{}: frame {}/{} ({:.1} FPS)",
                    tool_name, done, mock_frames, current_fps
                );
            }
            let secs = t0.elapsed().as_secs_f64();
            let final_fps = done as f64 / secs.max(0.001);
            let final_report = match err {
                Some(msg) => {
                    *prog_worker.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!("Cancelled: {msg}");
                    Err(msg)
                }
                None => {
                    *prog_worker.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!(
                        "Finished: processed {} frames in {:.2}s ({:.1} FPS)",
                        done, secs, final_fps
                    );
                    Ok(filmcraft_export::Report {
                        path: runner_out_path.clone(),
                        frames: done,
                        seconds: secs,
                        bytes: 0,
                        render_fps: final_fps,
                        extra_files: Vec::new(),
                    })
                }
            };
            *res_worker.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(final_report);
            prog_worker.finished.store(true, Ordering::Relaxed);
            return;
        }

        // Real execution with NCNN Vulkan worker
        let (script, param, bin) = match find_ai_assets() {
            Some(assets) => assets,
            None => {
                let err_msg = "AI assets not found (rvm_vulkan_worker.py or ncnn models)".to_string();
                *prog_worker.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = err_msg.clone();
                *res_worker.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Err(err_msg));
                prog_worker.finished.store(true, Ordering::Relaxed);
                return;
            }
        };

        *prog_worker.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = "Iniciando pipeline Vulkan GPU...".to_string();

        let mut cmd = std::process::Command::new("nice");
        cmd.args([
            "-n", "19",
            "ionice", "-c", "3",
            "uv", "run",
            "--with", "opencv-python-headless",
            "--with", "numpy",
            "--with", "ncnn",
            script.to_str().unwrap(),
            "-i", &input_path,
            "-o", &runner_out_path,
            "--param", param.to_str().unwrap(),
            "--model", bin.to_str().unwrap(),
            "-ss", &format!("{source_in_sec:.3}"),
            "-t", &format!("{duration_sec:.3}"),
            "--smoothing", &format!("{smoothing:.2}"),
            "--feather", &format!("{feather:.1}"),
            "--fps", &format!("{fps:.1}"),
            "--res", &res,
        ]);
        if !green_suppr {
            cmd.arg("--no-green-suppression");
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::null());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                let err_msg = format!("Failed to spawn AI worker process: {e}");
                *prog_worker.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = err_msg.clone();
                *res_worker.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Err(err_msg));
                prog_worker.finished.store(true, Ordering::Relaxed);
                return;
            }
        };

        let stdout = child.stdout.take().unwrap();
        let reader = std::io::BufReader::new(stdout);
        let mut last_done = 0u64;
        let mut cancelled = false;

        for line_res in reader.lines() {
            if prog_worker.cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                cancelled = true;
                break;
            }
            if let Ok(line) = line_res {
                parse_worker_line(&line, &prog_worker);
                last_done = prog_worker.done.load(Ordering::Relaxed);
            }
        }

        let status = child.wait();
        let secs = t0.elapsed().as_secs_f64();
        let final_fps = last_done as f64 / secs.max(0.001);

        let final_report = if cancelled {
            Err("Cancelled by user".to_string())
        } else if let Ok(st) = status {
            if st.success() {
                let bytes = std::fs::metadata(&runner_out_path).map(|m| m.len()).unwrap_or(0);
                Ok(filmcraft_export::Report {
                    path: runner_out_path.clone(),
                    frames: last_done,
                    seconds: secs,
                    bytes,
                    render_fps: final_fps,
                    extra_files: Vec::new(),
                })
            } else {
                Err(format!("AI worker exited with code {:?}", st.code()))
            }
        } else {
            Err("Failed to wait on worker process".to_string())
        };

        *res_worker.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(final_report);
        prog_worker.finished.store(true, Ordering::Relaxed);
    };

    let guarded_runner = crate::export_tools::guard_job(progress.clone(), result.clone(), runner);

    s.jobs.push(job);

    if wait || cfg!(target_arch = "wasm32") {
        guarded_runner();
        if auto_place {
            let res = result.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
            if let Some(Ok(rep)) = res {
                if !rep.path.is_empty() && std::path::Path::new(&rep.path).exists() {
                    let _ = s.execute("ai.apply", json!({
                        "path": rep.path,
                        "clip": clip_id.map(|c| c.0),
                    }));
                }
            }
        }
    } else {
        std::thread::Builder::new()
            .name(format!("ai-worker-{id}"))
            .spawn(guarded_runner)
            .map_err(|e| bad("ai.run", format!("failed to spawn background thread: {e}")))?;
    }

    Ok(json!({
        "job": id,
        "tool": tool_id,
        "clip": clip_id.map(|c| c.0),
        "total": detected_frames.max(1),
        "outputPath": output_path,
        "smoothing": smoothing,
        "feather": feather,
        "greenSuppression": green_suppr,
        "status": "started"
    }))
}

fn tool_status(s: &mut Session, p: &Value) -> Result<Value> {
    let id = u64_p(p, "job").ok_or_else(|| bad("ai.status", "need `job`"))?;
    let j = s.jobs.iter().find(|j| j.id == id).ok_or_else(|| bad("ai.status", format!("no job with id `{id}`")))?;

    let res = j.result.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
    let done = j.progress.done.load(Ordering::Relaxed);
    let total = j.progress.total.load(Ordering::Relaxed);
    let cancel = j.progress.cancel.load(Ordering::Relaxed);
    let finished = j.progress.finished.load(Ordering::Relaxed);
    let status_str = j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();

    Ok(json!({
        "job": id,
        "fraction": j.progress.fraction(),
        "done": done,
        "total": total,
        "etaSeconds": j.progress.eta().map(|d| d.as_secs_f64()),
        "status": status_str,
        "cancelled": cancel,
        "finished": finished,
        "result": match res {
            Some(Ok(r)) => serde_json::to_value(r).unwrap_or_default(),
            Some(Err(e)) => json!({"error": e}),
            None => Value::Null,
        }
    }))
}

fn cancel_tool(s: &mut Session, p: &Value) -> Result<Value> {
    let id = u64_p(p, "job").ok_or_else(|| bad("ai.cancel", "need `job`"))?;
    let j = s.jobs.iter().find(|j| j.id == id).ok_or_else(|| bad("ai.cancel", format!("no job with id `{id}`")))?;

    j.progress.cancel.store(true, Ordering::Relaxed);
    *j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = "Cancelled by user".to_string();

    Ok(json!({
        "job": id,
        "cancelled": true
    }))
}

fn apply_tool(s: &mut Session, p: &Value) -> Result<Value> {
    let path = str_p(p, "path").ok_or_else(|| bad("ai.apply", "need `path`"))?;
    let clip_id = u64_p(p, "clip").map(ClipId).or_else(|| {
        let q = s.active_sequence()?;
        s.state.selection.iter().copied().find(|c| {
            q.find_item(*c).is_some_and(|(t, _)| q.track(t).is_some_and(|trk| trk.kind == TrackKind::Video))
        })
    });

    // 1. Import media file into media pool
    let import_res = s.execute("file.import", json!({ "paths": [path] }))?;
    let items = import_res.get("items").and_then(Value::as_array).ok_or_else(|| bad("ai.apply", "import returned no items"))?;
    let item_id = items.first().and_then(Value::as_u64).ok_or_else(|| bad("ai.apply", "no item id from import"))?;

    // 2. Query timing and track from the original clip if present
    let (start, source_in, duration, effects_to_copy) = if let Some(cid) = clip_id {
        let q = s.active_sequence().ok_or_else(|| bad("ai.apply", "no active sequence"))?;
        let (_track_id, it) = q.find_item(cid).ok_or_else(|| bad("ai.apply", "original clip not found"))?;
        (it.start, it.source_in, it.duration, it.effects.clone())
    } else {
        (s.playhead(), filmcraft_time::Tick::ZERO, filmcraft_time::Tick::from_seconds_f64(5.0), Vec::new())
    };

    // Always place on the top-most track above all video layers
    let target_track = str_p(p, "track").map(str::to_string).unwrap_or_else(|| "new".to_string());

    // 3. Place clip on timeline
    let place_res = s.execute("timeline.place", json!({
        "item": item_id,
        "track": target_track,
        "time": start.0,
        "sourceIn": source_in.0,
        "duration": duration.0,
        "video": true,
        "audio": false,
        "insert": false,
    }))?;

    // 4. Inherit effects (motion position & scale) if available
    if !effects_to_copy.is_empty() {
        if let Some(placed_clip_id) = place_res.get("clip").and_then(Value::as_u64).map(ClipId).or_else(|| {
            place_res.get("clips").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_u64).map(ClipId)
        }) {
            if let Some(seq_id) = s.state.active_sequence {
                if let Some(q) = std::sync::Arc::make_mut(&mut s.project).sequence_mut(seq_id) {
                    if let Some((_, placed_item)) = q.find_item_mut(placed_clip_id) {
                        placed_item.effects = effects_to_copy;
                    }
                }
            }
        }
    }

    // Auto-save the project so changes are immediately persisted
    let _ = s.execute("file.save", json!({}));

    Ok(json!({
        "status": "applied",
        "importedItem": item_id,
        "track": target_track,
        "path": path,
        "placed": place_res
    }))
}

fn preview_tool(s: &mut Session, p: &Value) -> Result<Value> {
    let clip_id = u64_p(p, "clip").map(ClipId).or_else(|| {
        let q = s.active_sequence()?;
        s.state.selection.iter().copied().find(|c| {
            q.find_item(*c).is_some_and(|(t, _)| q.track(t).is_some_and(|trk| trk.kind == TrackKind::Video))
        })
    }).ok_or_else(|| bad("ai.preview", "no video clip selected"))?;

    let q = s.active_sequence().ok_or_else(|| bad("ai.preview", "no active sequence"))?;
    let (_, it) = q.find_item(clip_id).ok_or_else(|| bad("ai.preview", "clip not found in sequence"))?;
    let (_, media_clip, _) = s.project.resolve_media(it.item).ok_or_else(|| bad("ai.preview", "cannot resolve media for clip"))?;

    let media_path = match &media_clip.media {
        filmcraft_project::MediaRef::File { path } => path.clone(),
        _ => {
            let tmp_png = format!("/tmp/filmcraft_ai_preview_{}.png", clip_id.0);
            return Ok(json!({
                "ok": true,
                "path": tmp_png,
                "time": 0.0,
                "clip": clip_id.0,
                "mock": true
            }));
        }
    };

    let (script, param, bin) = find_ai_assets().ok_or_else(|| bad("ai.preview", "AI assets not found"))?;

    let time_sec = f64_p(p, "time").unwrap_or_else(|| {
        let playhead = s.playhead();
        if playhead >= it.start && playhead < it.start + it.duration {
            let offset = playhead - it.start;
            (it.source_in + offset).seconds()
        } else {
            it.source_in.seconds()
        }
    });

    let feather = f64_p(p, "feather").unwrap_or(2.0);
    let green_suppr = bool_p(p, "greenSuppression").unwrap_or(true);
    let res = str_p(p, "res").unwrap_or("480p");
    let out_png = format!("/tmp/filmcraft_ai_preview_{}.png", clip_id.0);

    let mut cmd = std::process::Command::new("nice");
    cmd.args([
        "-n", "19",
        "ionice", "-c", "3",
        "uv", "run",
        "--with", "opencv-python-headless",
        "--with", "numpy",
        "--with", "ncnn",
        script.to_str().unwrap(),
        "-i", &media_path,
        "-o", &out_png,
        "--param", param.to_str().unwrap(),
        "--model", bin.to_str().unwrap(),
        "--preview-time", &format!("{time_sec:.3}"),
        "--feather", &format!("{feather:.1}"),
        "--res", res,
    ]);
    if !green_suppr {
        cmd.arg("--no-green-suppression");
    }
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::null());

    let output = cmd.output().map_err(|e| bad("ai.preview", format!("failed to spawn worker: {e}")))?;
    if !output.status.success() {
        return Err(bad("ai.preview", "preview worker failed"));
    }

    Ok(json!({
        "ok": true,
        "path": out_png,
        "time": time_sec,
        "clip": clip_id.0
    }))
}
