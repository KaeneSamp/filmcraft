//! Unit tests for AI commands, job execution, status reporting, cancellation, and worker stream parsing.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde_json::json;

use crate::Session;
use crate::ai::parse_worker_line;

#[test]
fn ai_tools_list_returns_available_tools() {
    let mut s = Session::default();
    let res = s.execute("ai.tools", json!({})).unwrap();
    let tools = res["tools"].as_array().expect("tools array");
    assert!(!tools.is_empty());

    let rvm = tools.iter().find(|t| t["id"] == "matting.rvm").expect("matting.rvm tool");
    assert_eq!(rvm["name"], "Smart Matting (RVM Vulkan)");
    assert_eq!(rvm["category"], "Video");
    assert_eq!(rvm["accelerated"], true);
}

#[test]
fn ai_run_synchronous_mock_job_succeeds_and_updates_status() {
    let mut s = Session::default();
    let start_res = s
        .execute(
            "ai.run",
            json!({
                "tool": "matting.rvm",
                "wait": true,
                "mock": false,
                "mockFrames": 50,
                "smoothing": 0.85,
                "feather": 3.5,
                "greenSuppression": true
            }),
        )
        .unwrap();

    let job_id = start_res["job"].as_u64().expect("job id");
    assert_eq!(start_res["tool"], "matting.rvm");
    assert_eq!(start_res["total"], 50);

    // Query status
    let st = s.execute("ai.status", json!({"job": job_id})).unwrap();
    assert_eq!(st["job"], job_id);
    assert_eq!(st["done"], 50);
    assert_eq!(st["total"], 50);
    assert_eq!(st["finished"], true);
    assert_eq!(st["cancelled"], false);
    assert_eq!(st["fraction"], 1.0);

    // Result should contain frames and FPS
    let res = &st["result"];
    assert_eq!(res["frames"], 50);
    assert!(res["render_fps"].as_f64().unwrap_or(0.0) > 0.0);
}

#[test]
fn ai_cancel_stops_running_job() {
    let mut s = Session::default();
    let start_res = s
        .execute(
            "ai.run",
            json!({
                "tool": "matting.rvm",
                "wait": false,
                "mock": true,
                "mockFrames": 2000
            }),
        )
        .unwrap();

    let job_id = start_res["job"].as_u64().expect("job id");

    // Give it a tiny moment to run a few frames
    std::thread::sleep(std::time::Duration::from_millis(30));

    // Cancel the job
    let cancel_res = s.execute("ai.cancel", json!({"job": job_id})).unwrap();
    assert_eq!(cancel_res["cancelled"], true);

    // Wait briefly for background thread to exit loop
    std::thread::sleep(std::time::Duration::from_millis(50));

    let st = s.execute("ai.status", json!({"job": job_id})).unwrap();
    assert_eq!(st["cancelled"], true);
    assert_eq!(st["finished"], true);
}

#[test]
fn ai_worker_stream_parser_updates_progress() {
    let progress = Arc::new(filmcraft_export::Progress::default());

    // Feed lines
    let line1 = r#"{"total": 1200, "done": 100, "status": "Processing 100/1200"}"#;
    assert!(!parse_worker_line(line1, &progress));
    assert_eq!(progress.total.load(Ordering::Relaxed), 1200);
    assert_eq!(progress.done.load(Ordering::Relaxed), 100);
    assert_eq!(*progress.status.lock().unwrap(), "Processing 100/1200");

    let line2 = r#"{"done": 500, "status": "Processing 500/1200"}"#;
    assert!(!parse_worker_line(line2, &progress));
    assert_eq!(progress.done.load(Ordering::Relaxed), 500);

    let line_finish = r#"{"done": 1200, "status": "Complete", "finished": true}"#;
    assert!(parse_worker_line(line_finish, &progress));
    assert_eq!(progress.done.load(Ordering::Relaxed), 1200);
}

#[test]
fn ai_apply_imports_media_and_places_on_timeline() {
    let mut s = Session::default();
    s.execute("file.newSequence", json!({ "name": "Sequence 1", "video": 2, "audio": 2 })).unwrap();

    let test_mov = "/home/kaua/Videos/scrp/teste_worker_mov.mov";
    if std::path::Path::new(test_mov).exists() {
        let res = s.execute("ai.apply", json!({
            "path": test_mov,
            "track": "V2"
        })).unwrap();

        assert_eq!(res["status"], "applied");
        assert_eq!(res["track"], "V2");
        assert!(res["importedItem"].as_u64().is_some());

        let q = s.active_sequence().unwrap();
        let v2 = &q.video_tracks[1];
        assert_eq!(v2.items.len(), 1);
        assert_eq!(v2.items[0].item.0, res["importedItem"].as_u64().unwrap());
    }
}

#[test]
fn ai_preview_generates_frame_result() {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();

    let seq = s.execute("sequence.inspect", json!({})).unwrap();
    let first_clip = seq["video"][0]["items"][0]["clip"].as_u64().unwrap();

    let res = s.execute("ai.preview", json!({
        "clip": first_clip,
        "time": 0.5,
        "feather": 2.0,
        "greenSuppression": true,
    })).unwrap();

    assert_eq!(res["ok"], true);
    assert!(res["path"].as_str().is_some());
    assert_eq!(res["clip"], first_clip);
}

