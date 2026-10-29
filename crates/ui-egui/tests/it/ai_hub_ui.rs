//! Integration tests for the AI Studio panel: opening the panel from the Window menu,
//! inspecting registered automation controls, switching category tabs, and verifying docking.

use std::sync::mpsc::{Sender, channel};

use egui_kittest::Harness;
use filmcraft_engine::Session;
use filmcraft_ui_egui::FilmcraftApp;
use filmcraft_ui_egui::control::ControlRequest;
use filmcraft_ui_egui::dock::PanelKind;
use serde_json::{Value, json};

struct Driver {
    harness: Harness<'static, FilmcraftApp>,
    tx: Sender<ControlRequest>,
}

impl Driver {
    fn demo() -> Self {
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).expect("demo project");
        let (tx, rx) = channel();
        let app = FilmcraftApp::new(session).with_control(rx);
        let harness = Harness::builder().with_size(egui::vec2(1600.0, 980.0)).with_max_steps(10_000).build_eframe(move |_cc| app);
        let mut d = Driver { harness, tx };
        d.frames(4);
        d
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            let ctx = self.harness.ctx.clone();
            let mut raw = std::mem::take(self.harness.input_mut());
            eframe::App::raw_input_hook(self.harness.state_mut(), &ctx, &mut raw);
            *self.harness.input_mut() = raw;
            self.harness.step();
        }
    }

    fn call(&mut self, method: &str, params: Value) -> Value {
        let (req, reply) = ControlRequest::new(method, params.clone());
        self.tx.send(req).unwrap();
        for _ in 0..600 {
            self.frames(1);
            if let Ok(v) = reply.try_recv() {
                return v;
            }
        }
        panic!("no reply to {method} {params}");
    }

    fn ok(&mut self, method: &str, params: Value) -> Value {
        let v = self.call(method, params.clone());
        assert_eq!(v["ok"], json!(true), "{method} {params} failed: {v}");
        v["result"].clone()
    }

    fn exec(&mut self, command: &str, params: Value) -> Value {
        self.ok("engine.execute", json!({"command": command, "params": params}))
    }

    fn ids(&mut self, prefix: &str) -> Vec<String> {
        let v = self.ok("ui.elements", json!({"prefix": prefix}));
        v.as_array().unwrap().iter().filter_map(|e| e["id"].as_str().map(str::to_string)).collect()
    }

    fn click(&mut self, id: &str) {
        self.ok("ui.click", json!({"id": id}));
        self.frames(3);
    }

    fn state(&self) -> &FilmcraftApp {
        self.harness.state()
    }
}

#[test]
fn ai_studio_panel_opens_docks_and_exposes_controls() {
    let mut d = Driver::demo();

    // Verify AI Studio is in PanelKind::ALL
    assert!(PanelKind::ALL.contains(&PanelKind::AiHub));
    assert_eq!(PanelKind::AiHub.title(), "AI Studio");

    // Open via the Window menu command
    d.exec("window.panel.AiHub", json!({}));
    d.frames(4);

    // Verify the panel is open in the dock layout
    assert!(d.state().ui.dock.contains(&PanelKind::AiHub), "AI Studio panel must be docked");

    // Verify automation element IDs are present
    let ids = d.ids("ai.");
    assert!(ids.contains(&"ai.tab.Video".to_string()), "missing ai.tab.Video in {ids:?}");
    assert!(ids.contains(&"ai.tab.Audio".to_string()), "missing ai.tab.Audio in {ids:?}");
    assert!(ids.contains(&"ai.tab.Automation".to_string()), "missing ai.tab.Automation in {ids:?}");
    assert!(ids.contains(&"ai.run".to_string()), "missing ai.run in {ids:?}");

    // Inspect initial draft state
    assert_eq!(d.state().ui.ai.active_tab, "Video");
    assert_eq!(d.state().ui.ai.selected_tool, "matting.rvm");
    assert!((d.state().ui.ai.matting_smoothing - 0.70).abs() < f32::EPSILON);
    assert!((d.state().ui.ai.matting_feather - 2.0).abs() < f32::EPSILON);
}

#[test]
fn ai_studio_interactive_run_and_cancel_lifecycle() {
    let mut d = Driver::demo();
    d.exec("window.panel.AiHub", json!({}));
    d.frames(4);

    // Select the first video clip on the demo sequence
    let seq = d.exec("sequence.inspect", json!({}));
    let first_clip = seq["video"][0]["items"][0]["clip"].as_u64().unwrap();
    d.exec("timeline.select", json!({"clips": [first_clip]}));
    d.frames(4);

    // Verify clip selection is picked up and controls are rendered
    let ids = d.ids("ai.");
    assert!(ids.contains(&"ai.smoothing".to_string()), "missing ai.smoothing in {ids:?}");
    assert!(ids.contains(&"ai.feather".to_string()), "missing ai.feather in {ids:?}");
    assert!(ids.contains(&"ai.greenSuppression".to_string()), "missing ai.greenSuppression in {ids:?}");
    assert!(ids.contains(&"ai.preview.refresh".to_string()), "missing ai.preview.refresh in {ids:?}");
    assert!(ids.contains(&"ai.preview.overlay".to_string()), "missing ai.preview.overlay in {ids:?}");

    // Test preview refresh click
    d.click("ai.preview.refresh");
    d.frames(4);
    assert!(d.state().ui.ai.preview_path.is_some(), "preview_path should be populated after refresh");

    // Click Run button
    d.click("ai.run");
    d.frames(6);

    // A running job must be tracked in app.ui.ai.running_job
    let running_job = d.state().ui.ai.running_job;
    assert!(running_job.is_some(), "an AI job must have been launched");

    // The Cancel button must now be visible in automation elements
    let job_ids = d.ids("ai.");
    assert!(job_ids.contains(&"ai.cancel".to_string()), "ai.cancel must be visible during run: {job_ids:?}");

    // Click Cancel
    d.click("ai.cancel");
    d.frames(10);

    // Job should be finished/cancelled and ai.run button reappears
    let cancel_ids = d.ids("ai.");
    assert!(cancel_ids.contains(&"ai.run".to_string()), "ai.run button must reappear after cancel");
}

#[test]
fn ai_studio_silence_trimmer_tab_and_controls() {
    let mut d = Driver::demo();
    d.exec("window.panel.AiHub", json!({}));
    d.frames(4);

    // Click Automation tab
    d.click("ai.tab.Automation");
    d.frames(4);

    assert_eq!(d.state().ui.ai.active_tab, "Automation");
    assert_eq!(d.state().ui.ai.selected_tool, "timeline.silence");

    // Verify silence trimmer element IDs
    let ids = d.ids("ai.silence.");
    assert!(ids.contains(&"ai.silence.threshold".to_string()), "missing threshold slider: {ids:?}");
    assert!(ids.contains(&"ai.silence.min_duration".to_string()), "missing min_duration slider: {ids:?}");
    assert!(ids.contains(&"ai.silence.padding".to_string()), "missing padding slider: {ids:?}");
    assert!(ids.contains(&"ai.silence.all_tracks".to_string()), "missing all_tracks checkbox: {ids:?}");
    assert!(ids.contains(&"ai.silence.analyze".to_string()), "missing analyze button: {ids:?}");
    assert!(ids.contains(&"ai.silence.cut".to_string()), "missing cut button: {ids:?}");
}

