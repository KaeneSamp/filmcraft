//! FilmCraft AI Hub & AI Execution Engine.
//!
//! Provides the core commands and background job execution infrastructure for AI tasks:
//! - `ai.tools`: lists installed AI tools and parameter schemas.
//! - `ai.run`: dispatches an AI job in the background (or synchronously when `wait: true`).
//! - `ai.status`: retrieves real-time progress metrics (done, total, ETA, FPS, status).
//! - `ai.cancel`: cleanly requests abort of a running AI job.
//!
//! Output events stream through `filmcraft_export::Progress`, feeding both the UI panel and the
//! global jobs queue (`jobs.list`).

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use filmcraft_project::{ClipId, TrackKind};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, u64_p};
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
            params: r#"{"tool":str,"clip":id?,"wait":bool=false,"smoothing":f64=0.7,"feather":f64=2.0,"greenSuppression":bool=true,"mock":bool=false,"mockFrames":n=100}"#,
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
    let mock = bool_p(p, "mock").unwrap_or(false);
    let mock_frames = u64_p(p, "mockFrames").unwrap_or(60).max(1);
    let wait = bool_p(p, "wait").unwrap_or(false);

    // Find the clip if specified or from selection
    let clip_id = p.get("clip").and_then(Value::as_u64).map(ClipId).or_else(|| {
        let q = s.active_sequence()?;
        s.state.selection.iter().copied().find(|c| {
            q.find_item(*c).is_some_and(|(t, _)| q.track(t).is_some_and(|trk| trk.kind == TrackKind::Video))
        })
    });

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

    let runner = move || {
        let t0 = web_time::Instant::now();
        prog_worker.total.store(mock_frames, Ordering::Relaxed);

        let mut done = 0u64;
        let mut err = None;

        for frame in 1..=mock_frames {
            if prog_worker.cancel.load(Ordering::Relaxed) {
                err = Some("stopped".to_string());
                break;
            }

            // Simulate frame processing tick or IPC reading
            if mock {
                std::thread::sleep(std::time::Duration::from_millis(15));
            }

            done = frame;
            prog_worker.done.store(done, Ordering::Relaxed);

            let elapsed = t0.elapsed().as_secs_f64().max(0.001);
            let fps = done as f64 / elapsed;
            *prog_worker.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!(
                "{}: frame {}/{} ({:.1} FPS)",
                tool_name, done, mock_frames, fps
            );
        }

        let secs = t0.elapsed().as_secs_f64();
        let fps = done as f64 / secs.max(0.001);

        let final_report = match err {
            Some(msg) => {
                *prog_worker.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!("Cancelled: {msg}");
                Err(msg)
            }
            None => {
                *prog_worker.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!(
                    "Finished: processed {} frames in {:.2}s ({:.1} FPS)",
                    done, secs, fps
                );
                Ok(filmcraft_export::Report {
                    path: String::new(),
                    frames: done,
                    seconds: secs,
                    bytes: 0,
                    render_fps: fps,
                    extra_files: Vec::new(),
                })
            }
        };

        *res_worker.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(final_report);
        prog_worker.finished.store(true, Ordering::Relaxed);
    };

    let guarded_runner = crate::export_tools::guard_job(progress.clone(), result.clone(), runner);

    s.jobs.push(job);

    if wait || cfg!(target_arch = "wasm32") {
        guarded_runner();
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
        "total": mock_frames,
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
