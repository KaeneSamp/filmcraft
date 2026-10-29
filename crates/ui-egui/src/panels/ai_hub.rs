//! AI Studio panel: interactive suite for local and Vulkan AI tools directly in FilmCraft.
//! Empowers editors to run neural background matting (RVM Vulkan), automated captions (Whisper),
//! vocal isolation (Demucs), and custom AI pipelines without UI freezes or external terminal scripts.
//!
//! Automation ids: `ai.tab.<Video|Audio|Automation>`, `ai.tool.<Id>`, `ai.run`, `ai.cancel`,
//! `ai.feather`, `ai.smoothing`, `ai.greenSuppression`.

use std::sync::atomic::Ordering;

use egui::{Align2, Color32, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use filmcraft_project::{ClipId, TrackItem, TrackKind};
use filmcraft_time::{TimeDisplay, format_time};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::FilmcraftApp;
use crate::icons::{self, Icon};
use crate::theme::Tokens;

const PAD: f32 = 14.0;
const FOOTER_H: f32 = 68.0;

/// Persistent draft / UI state of the AI Studio panel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AiHubState {
    /// Active category tab ("Video", "Audio", "Automation").
    pub active_tab: String,
    /// Identifier of the selected AI tool.
    pub selected_tool: String,
    /// Feathering blur radius for mask borders (in pixels: 0.0 .. 8.0).
    pub matting_feather: f32,
    /// Temporal EMA smoothing factor (0.0 .. 1.0, e.g. 0.70).
    pub matting_smoothing: f32,
    /// Soft green / background suppression toggle.
    pub matting_green_suppression: bool,
    /// Selected output codec / format ("prores4444", "png_seq").
    pub output_format: String,
    /// Active job id when an AI background task is running.
    pub running_job: Option<u64>,
    /// Live user status message.
    pub status_message: String,
}

impl Default for AiHubState {
    fn default() -> Self {
        Self {
            active_tab: "Video".to_string(),
            selected_tool: "matting.rvm".to_string(),
            matting_feather: 2.0,
            matting_smoothing: 0.70,
            matting_green_suppression: true,
            output_format: "prores4444".to_string(),
            running_job: None,
            status_message: String::new(),
        }
    }
}

/// Helper to get the primary selected video clip on the active sequence with its duration and track label.
pub fn selected_video_clip(app: &FilmcraftApp) -> Option<(ClipId, TrackItem, String, String)> {
    let seq = app.session.active_sequence()?;
    for &cid in &app.session.state.selection {
        if let Some((tid, item)) = seq.find_item(cid) {
            if let Some(track) = seq.track(tid) {
                if track.kind == TrackKind::Video {
                    let name = app
                        .session
                        .project
                        .items
                        .get(&item.item)
                        .map(|m| m.name.clone())
                        .unwrap_or_else(|| item.name.clone());

                    let tc = format_time(
                        item.duration,
                        seq.settings.frame_rate,
                        seq.settings.drop_frame,
                        TimeDisplay::Timecode,
                        48_000,
                    );
                    return Some((cid, item.clone(), name, tc));
                }
            }
        }
    }
    None
}

/// Render the AI Studio panel.
pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    let painter = ui.painter().clone();
    painter.rect_filled(rect, 0.0, t.panel_bg);

    let active_clip = selected_video_clip(app);
    let auto = &mut app.auto;

    // Check if a background job is currently running
    let mut finished_job_report = None;
    let active_job_data = if let Some(job_id) = app.ui.ai.running_job {
        if let Some(j) = app.session.jobs.iter().find(|j| j.id == job_id) {
            let frac = j.progress.fraction();
            let status = j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
            let is_finished = j.progress.finished.load(Ordering::Relaxed);
            let eta = j.progress.eta();
            let done = j.progress.done.load(Ordering::Relaxed);
            let total = j.progress.total.load(Ordering::Relaxed);

            if is_finished {
                let res = j.result.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
                finished_job_report = Some(res);
                None
            } else {
                Some((job_id, frac, status, done, total, eta))
            }
        } else {
            None
        }
    } else {
        None
    };

    if let Some(report_res) = finished_job_report {
        app.ui.ai.running_job = None;
        match report_res {
            Some(Ok(rep)) => {
                if !rep.path.is_empty() {
                    let clip_opt = active_clip.as_ref().map(|(c, _, _, _)| c.0);
                    let _ = app.session.execute("ai.apply", json!({
                        "path": rep.path,
                        "clip": clip_opt,
                    }));
                }
                app.ui.status = format!("IA concluída: {} quadros processados em {:.1}s!", rep.frames, rep.seconds);
                app.ui.ai.status_message = format!("Concluído: {} quadros em {:.1}s ({:.1} FPS)", rep.frames, rep.seconds, rep.render_fps);
            }
            Some(Err(e)) => {
                app.ui.status = format!("Falha no processamento: {e}");
                app.ui.ai.status_message = format!("Erro: {e}");
            }
            None => {
                app.ui.ai.status_message = "Processo finalizado.".to_string();
            }
        }
    }

    // Header strip (Top 40px)
    let header_rect = Rect::from_min_size(rect.min, vec2(rect.width(), 40.0));
    painter.rect_filled(header_rect, 0.0, t.field_bg);
    painter.line_segment(
        [pos2(header_rect.min.x, header_rect.max.y), pos2(header_rect.max.x, header_rect.max.y)],
        Stroke::new(1.0, t.separator),
    );

    // Header Icon (Sparkle) + Title
    let icon_rect = Rect::from_min_size(pos2(header_rect.min.x + PAD, header_rect.min.y + 12.0), vec2(16.0, 16.0));
    icons::paint(&painter, icon_rect, Icon::Sparkle, t.accent);

    painter.text(
        pos2(icon_rect.max.x + 8.0, header_rect.center().y),
        Align2::LEFT_CENTER,
        "AI Studio",
        Tokens::ui(13.0),
        t.text,
    );

    // Hardware acceleration badge
    let badge_text = "Vulkan GPU";
    let badge_font = Tokens::ui(10.5);
    let badge_layout = painter.layout_no_wrap(badge_text.to_string(), badge_font.clone(), t.accent);
    let badge_rect = Rect::from_min_size(
        pos2(header_rect.max.x - PAD - badge_layout.size().x - 12.0, header_rect.center().y - 9.0),
        vec2(badge_layout.size().x + 12.0, 18.0),
    );
    painter.rect_filled(badge_rect, 9.0, t.panel_bg);
    painter.rect_stroke(badge_rect, 9.0, Stroke::new(1.0, t.accent), StrokeKind::Inside);
    painter.text(badge_rect.center(), Align2::CENTER_CENTER, badge_text, badge_font, t.accent);

    // Category Tabs: Video | Audio | Automation
    let tabs_y = header_rect.max.y + 12.0;
    let mut tab_x = rect.min.x + PAD;
    for tab_name in ["Video", "Audio", "Automation"] {
        let is_selected = app.ui.ai.active_tab == tab_name;
        let text_layout = painter.layout_no_wrap(tab_name.to_string(), Tokens::ui(12.0), t.text);
        let tab_rect = Rect::from_min_size(pos2(tab_x, tabs_y - 8.0), vec2(text_layout.size().x + 4.0, 22.0));
        let resp = ui.interact(tab_rect, egui::Id::unique(("ai-tab", tab_name)), Sense::click());
        let col = if is_selected || resp.hovered() { t.text } else { t.text_dim };

        painter.text(pos2(tab_x, tabs_y + 2.0), Align2::LEFT_CENTER, tab_name, Tokens::ui(12.0), col);
        if is_selected {
            painter.line_segment(
                [pos2(tab_x, tabs_y + 12.0), pos2(tab_x + text_layout.size().x, tabs_y + 12.0)],
                Stroke::new(2.0, t.accent),
            );
        }
        auto.add(&format!("ai.tab.{tab_name}"), tab_rect, tab_name);

        if resp.clicked() {
            app.ui.ai.active_tab = tab_name.to_string();
        }
        tab_x += text_layout.size().x + 20.0;
    }

    // Scrollable Main Body
    let body_top = tabs_y + 18.0;
    let body_rect = Rect::from_min_max(
        pos2(rect.min.x + PAD, body_top),
        pos2(rect.max.x - PAD, rect.max.y - FOOTER_H),
    );

    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body_rect).id_salt("ai-hub-body"));
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(&mut child, |ui| {
            ui.set_max_width(ui.available_width() - 4.0);
            let w = ui.available_width();

            match app.ui.ai.active_tab.as_str() {
                "Video" => {
                    // Selected Clip Target Card
                    egui::Frame::new()
                        .fill(t.field_bg)
                        .stroke(Stroke::new(1.0, t.field_border))
                        .corner_radius(6.0)
                        .inner_margin(10.0)
                        .show(ui, |ui| {
                            ui.set_width(w - 20.0);
                            match &active_clip {
                                Some((cid, _item, name, tc)) => {
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("🎯 Clipe Selecionado:").size(12.0).color(t.text_dim));
                                        ui.label(egui::RichText::new(name).size(12.5).strong().color(t.text));
                                    });
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("ID #{cid:?} | Duração: {tc}"))
                                                .size(11.0)
                                                .color(t.text_faint),
                                        );
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            ui.label(egui::RichText::new("● Pronto").size(11.0).color(Color32::from_rgb(0x40, 0xc0, 0x57)));
                                        });
                                    });
                                }
                                None => {
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new("ℹ️").size(13.0));
                                        ui.label(
                                            egui::RichText::new("Nenhum clipe de vídeo selecionado na timeline.")
                                                .size(11.5)
                                                .color(t.text_dim),
                                        );
                                    });
                                    ui.label(
                                        egui::RichText::new("Clique no clipe da webcam na timeline para habilitar o processamento.")
                                            .size(10.5)
                                            .color(t.text_faint),
                                    );
                                }
                            }
                        });

                    ui.add_space(12.0);

                    // Tool Info Card
                    ui.label(egui::RichText::new("Ferramenta Ativa").size(11.0).color(t.text_faint));
                    ui.label(egui::RichText::new("Smart Matting (RVM Vulkan)").size(13.5).strong().color(t.text));
                    ui.label(
                        egui::RichText::new("Recorte de silhueta por IA em tempo real com preservação de roupas pretas, cabelos e canal alfa nativo 12-bit.")
                            .size(11.0)
                            .color(t.text_dim),
                    );

                    ui.add_space(14.0);
                    ui.separator();
                    ui.add_space(10.0);

                    // Fine-Tuning Parameters
                    ui.label(egui::RichText::new("Parâmetros do Recorte").size(12.0).strong().color(t.text));
                    ui.add_space(6.0);

                    // 1. Suavização Temporal (EMA)
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Suavização Temporal").size(11.5).color(t.text));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(egui::RichText::new(format!("{:.0}%", app.ui.ai.matting_smoothing * 100.0)).size(11.5).color(t.text_dim));
                        });
                    });
                    let s_resp = ui.add(egui::Slider::new(&mut app.ui.ai.matting_smoothing, 0.0..=1.0).show_value(false));
                    auto.add("ai.smoothing", s_resp.rect, "Suavização Temporal");
                    ui.label(egui::RichText::new("Elimina cintilação e oscilações de contorno entre os quadros.").size(10.0).color(t.text_faint));

                    ui.add_space(10.0);

                    // 2. Feathering / Anti-aliasing de Borda
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Suavidade de Contorno (Feathering)").size(11.5).color(t.text));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(egui::RichText::new(format!("{:.1} px", app.ui.ai.matting_feather)).size(11.5).color(t.text_dim));
                        });
                    });
                    let f_resp = ui.add(egui::Slider::new(&mut app.ui.ai.matting_feather, 0.0..=8.0).show_value(false));
                    auto.add("ai.feather", f_resp.rect, "Feathering do Alfa");
                    ui.label(egui::RichText::new("Transição suave de subpixel sem corte serrilhado.").size(10.0).color(t.text_faint));

                    ui.add_space(10.0);

                    // 3. Supressão de Verde / Despeckle
                    let g_resp = ui.checkbox(&mut app.ui.ai.matting_green_suppression, "Atenuar reflexos e ruído de fundo");
                    auto.add("ai.greenSuppression", g_resp.rect, "Supressão de Verde");
                    ui.add_space(10.0);

                    // 4. Formato de Saída
                    ui.label(egui::RichText::new("Codec do Arquivo Transparente").size(11.5).color(t.text));
                    let c_resp = egui::Frame::new()
                        .fill(t.field_bg)
                        .stroke(Stroke::new(1.0, t.field_border))
                        .corner_radius(4.0)
                        .inner_margin(6.0)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Apple ProRes 4444 (YUVA 4:4:4 12-bit)").size(11.5).color(t.text));
                        });
                    auto.add("ai.codec", c_resp.response.rect, "Codec de Saída");
                }
                "Audio" => {
                    ui.label(egui::RichText::new("Módulos de Áudio com IA").size(13.0).strong().color(t.text));
                    ui.add_space(8.0);
                    for (id, name, desc) in [
                        ("audio.demucs", "Isolamento Vocal (Demucs)", "Remove música e ruídos de fundo da voz do streamer."),
                        ("caption.whisper", "Auto-Legendas (Whisper Local)", "Gera legendas sincronizadas palavra por palavra na timeline."),
                        ("audio.denoise", "Denoise de Estúdio", "Limpeza acústica em tempo real via modelos neurais."),
                    ] {
                        let is_active = app.ui.ai.selected_tool == id;
                        egui::Frame::new()
                            .fill(if is_active { t.pill_active_bg } else { t.field_bg })
                            .stroke(Stroke::new(1.0, if is_active { t.accent } else { t.field_border }))
                            .corner_radius(6.0)
                            .inner_margin(8.0)
                            .show(ui, |ui| {
                                ui.set_width(w - 16.0);
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new(name).size(12.0).strong().color(t.text));
                                    if is_active {
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            ui.label(egui::RichText::new("Ativo").size(10.5).color(t.accent));
                                        });
                                    }
                                });
                                ui.label(egui::RichText::new(desc).size(11.0).color(t.text_dim));
                            });
                        auto.add(&format!("ai.tool.{id}"), ui.min_rect(), name);
                        ui.add_space(6.0);
                    }
                }
                _ => {
                    ui.label(egui::RichText::new("Automações Inteligentes").size(13.0).strong().color(t.text));
                    ui.add_space(8.0);
                    for (id, name, desc) in [
                        ("scene.detect", "Detecção de Cortes de Cena", "Detecta mudanças de plano e cria cortes automáticos na timeline."),
                        ("timeline.silence", "Aparador de Silêncios", "Localiza pausas longas e acelera o ritmo da gravação."),
                    ] {
                        egui::Frame::new()
                            .fill(t.field_bg)
                            .stroke(Stroke::new(1.0, t.field_border))
                            .corner_radius(6.0)
                            .inner_margin(8.0)
                            .show(ui, |ui| {
                                ui.set_width(w - 16.0);
                                ui.label(egui::RichText::new(name).size(12.0).strong().color(t.text));
                                ui.label(egui::RichText::new(desc).size(11.0).color(t.text_dim));
                            });
                        auto.add(&format!("ai.tool.{id}"), ui.min_rect(), name);
                        ui.add_space(6.0);
                    }
                }
            }
        });

    // Action Footer
    let footer_rect = Rect::from_min_max(
        pos2(rect.min.x, rect.max.y - FOOTER_H),
        rect.max,
    );
    painter.rect_filled(footer_rect, 0.0, t.field_bg);
    painter.line_segment(
        [pos2(footer_rect.min.x, footer_rect.min.y), pos2(footer_rect.max.x, footer_rect.min.y)],
        Stroke::new(1.0, t.separator),
    );

    let mut footer_ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(footer_rect.shrink2(vec2(PAD, 8.0)))
            .id_salt("ai-hub-footer"),
    );

    // If an AI job is currently running, show live progress bar and Cancel button
    if let Some((job_id, frac, status, done, total, eta)) = active_job_data {
        footer_ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("⚡ {}", status)).size(11.0).color(t.text));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(d) = eta {
                        ui.label(egui::RichText::new(format!("ETA {:02}:{:02}", d.as_secs() / 60, d.as_secs() % 60)).size(11.0).color(t.text_faint));
                    }
                    ui.label(egui::RichText::new(format!("{done}/{total}")).size(11.0).color(t.text_dim));
                });
            });
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.add(egui::ProgressBar::new(frac).desired_width(ui.available_width() - 85.0).show_percentage());
                let cancel_btn = egui::Button::new(egui::RichText::new("Cancelar").size(11.5).color(Color32::WHITE)).fill(t.danger);
                let resp = ui.add(cancel_btn);
                auto.add("ai.cancel", resp.rect, "Cancelar IA");
                if resp.clicked() {
                    let _ = app.session.execute("ai.cancel", json!({"job": job_id}));
                }
            });
        });
    } else {
        // Idle footer: Primary run CTA button
        footer_ui.horizontal(|ui| {
            let has_clip = active_clip.is_some();
            let btn_label = if has_clip {
                "✨ Processar Remoção de Fundo"
            } else {
                "Selecione um clipe na timeline"
            };

            let btn = egui::Button::new(
                egui::RichText::new(btn_label)
                    .size(12.5)
                    .color(if has_clip { Color32::WHITE } else { t.text_dim }),
            )
            .fill(if has_clip { t.accent } else { t.panel_bg })
            .min_size(vec2(ui.available_width(), 36.0));

            let resp = ui.add_enabled(has_clip, btn);
            auto.add("ai.run", resp.rect, "Executar IA");

            if resp.clicked() {
                let has_assets = filmcraft_engine::ai::find_ai_assets().is_some();
                let params = json!({
                    "tool": app.ui.ai.selected_tool,
                    "clip": active_clip.as_ref().map(|(c, _, _, _)| c.0),
                    "smoothing": app.ui.ai.matting_smoothing,
                    "feather": app.ui.ai.matting_feather,
                    "greenSuppression": app.ui.ai.matting_green_suppression,
                    "mock": !has_assets,
                    "mockFrames": 120,
                });
                match app.session.execute("ai.run", params) {
                    Ok(r) => {
                        if let Some(id) = r.get("job").and_then(serde_json::Value::as_u64) {
                            app.ui.ai.running_job = Some(id);
                            app.ui.status = if has_assets {
                                "Iniciando processamento com IA Vulkan (GPU)...".to_string()
                            } else {
                                "Iniciando simulação de teste...".to_string()
                            };
                        }
                    }
                    Err(e) => {
                        app.ui.status = format!("Erro ao disparar IA: {e}");
                    }
                }
            }
        });
    }
}
