# dashboard.py — OpenCV frame annotation and all Streamlit UI components.

import time
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from config import NUM_SIGNALS, SIGNAL_STATE_COLOR, YELLOW_SEC, SignalState, SystemStats

SIGNAL_STATE_COLOR_EXT = {
    **SIGNAL_STATE_COLOR,
    "ALL_RED": (200, 200, 200),
}


# ── OpenCV frame annotation ───────────────────────────────────────────────────

def draw_dashboard(
    frame: np.ndarray,
    signals: list[SignalState],
    zones: list[tuple[int, int, int, int]],
    stats: SystemStats,
    is_image: bool = False,
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    for i, (zx1, zy1, zx2, zy2) in enumerate(zones):
        sig   = signals[i]
        color = SIGNAL_STATE_COLOR_EXT.get(sig.state, SIGNAL_STATE_COLOR_EXT["RED"])
        bgr   = (color[2], color[1], color[0])

        alpha   = 0.06 if sig.state in ("RED", "ALL_RED") else 0.18
        overlay = vis.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), bgr, -1)
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        cv2.rectangle(vis, (zx1, zy1), (zx2, zy2), bgr,
                      3 if sig.state not in ("RED", "ALL_RED") else 1)

        # Signal name pill
        px, py = zx1 + 6, 6
        cv2.rectangle(vis, (px, py), (px + 52, py + 28), bgr, -1)
        cv2.putText(vis, sig.name, (px + 6, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        if sig.skipped:
            cv2.putText(vis, "SKIP", (px + 56, py + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (250, 204, 21), 1, cv2.LINE_AA)

        # Vehicle count badge
        bx, by = zx1 + 6, zy2 - 40
        cv2.rectangle(vis, (bx, by), (bx + 44, by + 28), (20, 20, 30), -1)
        cv2.putText(vis, str(sig.vehicle_count), (bx + 6, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr, 2, cv2.LINE_AA)

        # Wait time on RED
        if sig.state == "RED" and sig.wait_time > 2:
            cv2.putText(vis, f"{sig.wait_time:.0f}s", (bx + 50, by + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (239, 68, 68), 1, cv2.LINE_AA)

        # Bounding boxes
        for det in sig.detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), bgr, 2)
            label = f"{det['class_name']} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    if is_image:
        bar = f"Inference: {1000/max(stats.fps,1e-6):.0f}ms | Vehicles: {stats.total_vehicles}"
    else:
        bar = (f"FPS:{stats.fps:.1f} | Vehicles:{stats.total_vehicles} | "
               f"Cycle:{stats.cycle_count} | AvgWait:{stats.avg_wait_time:.1f}s")
    cv2.rectangle(vis, (0, h - 34), (w, h), (10, 10, 20), -1)
    cv2.putText(vis, bar, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1, cv2.LINE_AA)
    return vis


# ── HTML helpers ──────────────────────────────────────────────────────────────

def traffic_badge_html(level: str) -> str:
    palette = {
        "LOW":    ("#bbf7d0", "#166534"),
        "MEDIUM": ("#fef9c3", "#854d0e"),
        "HIGH":   ("#fecaca", "#991b1b"),
    }
    bg, fg = palette.get(level, ("#e2e8f0", "#334155"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:12px;font-size:0.75rem;font-weight:700;">{level}</span>')


def _bar_color(pct: float) -> str:
    if pct > 50: return "#22c55e"
    if pct > 20: return "#facc15"
    return "#ef4444"


# ── Signal panel ──────────────────────────────────────────────────────────────

def render_full_signal_panel(signals: list[SignalState], controller=None) -> None:
    """
    4-signal panel. Each card shows:
      GREEN  — live draining countdown bar + remaining seconds
      YELLOW — yellow countdown (YELLOW_SEC → 0)
      RED    — wait time climbing up in red
      ALL_RED— brief white flash
    """
    cols = st.columns(NUM_SIGNALS)
    now  = time.time()

    for i, col in enumerate(cols):
        sig        = signals[i]
        is_green   = sig.state == "GREEN"
        is_yellow  = sig.state == "YELLOW"
        is_allred  = sig.state == "ALL_RED"
        is_red     = sig.state == "RED"

        # Bulb colours
        red_clr    = "#ef4444" if (is_red or is_allred) else "#3b0d0d"
        yellow_clr = "#facc15" if is_yellow             else "#3b2f00"
        green_clr  = "#22c55e" if is_green              else "#0a2e18"

        red_glow    = "box-shadow:0 0 18px 6px #ef4444;" if (is_red or is_allred) else ""
        yellow_glow = "box-shadow:0 0 18px 6px #facc15;" if is_yellow             else ""
        green_glow  = "box-shadow:0 0 18px 6px #22c55e;" if is_green              else ""

        border_clr = ("#22c55e" if is_green else
                      "#facc15" if is_yellow else
                      "#c0c0c0" if is_allred else "#1e293b")

        # ── Timer block ───────────────────────────────────────────────────────
        timer_html = ""

        if is_green and controller is not None:
            remaining = controller.remaining
            total     = max(controller.green_duration, 1)
            pct       = (remaining / total) * 100
            bar_clr   = _bar_color(pct)
            timer_html = f"""
            <div style="margin-top:10px;padding:0 6px;">
              <div style="display:flex;justify-content:space-between;
                          font-size:0.7rem;color:#64748b;margin-bottom:4px;">
                <span>🟢 GREEN</span>
                <span style="color:{bar_clr};font-weight:800;font-size:1rem;">
                  {remaining:.1f}s
                </span>
              </div>
              <div style="background:#1e293b;border-radius:6px;height:10px;overflow:hidden;">
                <div style="width:{pct:.1f}%;height:100%;background:{bar_clr};
                            border-radius:6px;transition:width 0.4s linear;"></div>
              </div>
              <div style="font-size:0.62rem;color:#475569;margin-top:3px;
                          display:flex;justify-content:space-between;">
                <span>0s</span><span>allotted {total:.0f}s</span>
              </div>
            </div>"""

        elif is_yellow and controller is not None:
            y_rem   = controller.yellow_remaining
            y_pct   = (y_rem / max(YELLOW_SEC, 0.1)) * 100
            timer_html = f"""
            <div style="margin-top:10px;padding:0 6px;">
              <div style="display:flex;justify-content:space-between;
                          font-size:0.7rem;color:#64748b;margin-bottom:4px;">
                <span>🟡 YELLOW</span>
                <span style="color:#facc15;font-weight:800;font-size:1rem;">
                  {y_rem:.1f}s
                </span>
              </div>
              <div style="background:#1e293b;border-radius:6px;height:10px;overflow:hidden;">
                <div style="width:{y_pct:.1f}%;height:100%;background:#facc15;
                            border-radius:6px;transition:width 0.2s linear;"></div>
              </div>
            </div>"""

        elif is_allred:
            timer_html = """
            <div style="margin-top:10px;font-size:0.72rem;color:#c0c0c0;
                        font-weight:700;letter-spacing:1px;text-align:center;">
              ⬛ ALL RED — CLEARING
            </div>"""

        elif is_red:
            wait = sig.wait_time
            timer_html = f"""
            <div style="margin-top:10px;padding:0 6px;">
              <div style="display:flex;justify-content:space-between;
                          font-size:0.7rem;color:#64748b;margin-bottom:4px;">
                <span>🔴 waiting</span>
                <span style="color:#ef4444;font-weight:800;font-size:1rem;">
                  {wait:.0f}s
                </span>
              </div>
              <div style="background:#1e293b;border-radius:6px;height:6px;overflow:hidden;">
                <div style="width:100%;height:100%;background:#3b0d0d;border-radius:6px;">
                </div>
              </div>
            </div>"""

        with col:
            st.markdown(
                f"""
                <div style="background:#0f172a;border-radius:16px;padding:14px 8px;
                            text-align:center;border:2px solid {border_clr};
                            min-height:280px;">
                  <div style="font-size:1rem;font-weight:700;color:#94a3b8;
                              margin-bottom:10px;">{sig.name}</div>
                  <div style="display:inline-block;background:#1e293b;border-radius:12px;
                              padding:10px 14px;border:2px solid #334155;">
                    <div style="width:36px;height:36px;border-radius:50%;
                                background:{red_clr};{red_glow}margin:0 auto 8px;"></div>
                    <div style="width:36px;height:36px;border-radius:50%;
                                background:{yellow_clr};{yellow_glow}margin:0 auto 8px;"></div>
                    <div style="width:36px;height:36px;border-radius:50%;
                                background:{green_clr};{green_glow}margin:0 auto;"></div>
                  </div>
                  <div style="margin-top:10px;font-size:1.4rem;font-weight:800;
                              color:#f1f5f9;">{sig.vehicle_count}</div>
                  <div style="font-size:0.68rem;color:#64748b;">vehicles</div>
                  <div style="margin-top:5px;">{traffic_badge_html(sig.traffic_level)}</div>
                  {timer_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Metrics ───────────────────────────────────────────────────────────────────

def render_metric_bar(stats: SystemStats) -> None:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🚦 Active",    f"S{stats.active_signal}", stats.active_state)
    m2.metric("🚗 Vehicles",  stats.total_vehicles)
    m3.metric("📡 FPS",       f"{stats.fps:.1f}")
    m4.metric("🔄 Cycles",    stats.cycle_count)
    m5.metric("⏭️ Skipped",   stats.skipped_count)
    m6.metric("⏱️ Avg Wait",  f"{stats.avg_wait_time:.1f}s")


# ── Analytics sidebar ─────────────────────────────────────────────────────────

def render_analytics_sidebar(signals: list[SignalState]) -> None:
    for sig in signals:
        icon = ("🟢" if sig.state == "GREEN" else
                "🟡" if sig.state == "YELLOW" else
                "⬛" if sig.state == "ALL_RED" else "🔴")
        st.markdown(
            f"{icon} **{sig.name}** — {sig.vehicle_count} veh "
            f"| {traffic_badge_html(sig.traffic_level)}",
            unsafe_allow_html=True,
        )


# ── History ───────────────────────────────────────────────────────────────────

def update_history(history: dict, signals: list[SignalState], max_len: int = 60) -> dict:
    for sig in signals:
        history[sig.name].append(sig.vehicle_count)
        if len(history[sig.name]) > max_len:
            history[sig.name].pop(0)
    return history


def render_chart(chart_placeholder, history: dict) -> None:
    chart_data = {k: v for k, v in history.items() if v}
    if not chart_data:
        return
    max_len = max(len(v) for v in chart_data.values())
    padded  = {k: ([0] * (max_len - len(v))) + v for k, v in chart_data.items()}
    chart_placeholder.line_chart(pd.DataFrame(padded), height=160)


# ── Control panel ─────────────────────────────────────────────────────────────

def render_control_panel(controller) -> dict:
    """
    AUTO/MANUAL toggle + live timing info + Force Next Signal button.
    Returns: { 'mode': str|None, 'set_signal': int|None, 'force_next': bool }
    """
    action    = {"mode": None, "set_signal": None, "force_next": False}
    is_manual = controller.manual_mode

    badge_bg  = "#7f1d1d" if is_manual else "#14532d"
    badge_fg  = "#ef4444" if is_manual else "#22c55e"
    mode_lbl  = "🔴 MANUAL" if is_manual else "🟢 AUTO"

    st.markdown(
        f"""
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:14px;
                    padding:14px 18px;margin-bottom:10px;">
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <span style="font-size:1rem;font-weight:700;color:#94a3b8;">
              🎛️ Signal Control Mode
            </span>
            <span style="background:{badge_bg};color:{badge_fg};
                         padding:3px 14px;border-radius:20px;
                         font-size:0.8rem;font-weight:700;">{mode_lbl}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🤖 Auto Mode", use_container_width=True,
                     type="primary" if not is_manual else "secondary"):
            action["mode"] = "AUTO"
    with c2:
        if st.button("🕹️ Manual Mode", use_container_width=True,
                     type="primary" if is_manual else "secondary"):
            action["mode"] = "MANUAL"

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    # ── AUTO timing panel ─────────────────────────────────────────────────────
    if not is_manual:
        active    = controller.signals[controller.active_idx]
        remaining = controller.remaining
        total     = max(controller.green_duration, 1)
        pct       = (remaining / total) * 100
        bar_clr   = _bar_color(pct)
        phase_clr = "#22c55e" if controller.phase == "GREEN" else "#facc15"

        # Build per-signal timing preview
        signal_rows = ""
        for sig in controller.signals:
            s_icon = ("🟢" if sig.state == "GREEN" else
                      "🟡" if sig.state == "YELLOW" else
                      "⬛" if sig.state == "ALL_RED" else "🔴")
            if sig.state == "GREEN":
                t_str = f"<b style='color:{bar_clr};'>{remaining:.1f}s left</b>"
            elif sig.state == "YELLOW":
                t_str = f"<b style='color:#facc15;'>{controller.yellow_remaining:.1f}s</b>"
            elif sig.state == "RED":
                t_str = f"<span style='color:#ef4444;'>wait {sig.wait_time:.0f}s</span>"
            else:
                t_str = "<span style='color:#c0c0c0;'>clearing</span>"
            signal_rows += (
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:3px 0;border-bottom:1px solid #1e293b;'>"
                f"<span style='color:#94a3b8;font-size:0.75rem;'>"
                f"{s_icon} {sig.name} — {sig.vehicle_count} veh</span>"
                f"<span style='font-size:0.75rem;'>{t_str}</span></div>"
            )

        st.markdown(
            f"""
            <div style="background:#0c1a2e;border:1px solid #1e3a5f;
                        border-radius:12px;padding:12px 16px;margin-bottom:10px;">
              <div style="color:#60a5fa;font-size:0.78rem;font-weight:600;
                          margin-bottom:8px;">⚙️ AUTO — Adaptive Corridor Timing</div>
              <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;">
                <span style="color:#94a3b8;font-size:0.75rem;">
                  Active: <b style="color:#f1f5f9;">{active.name}</b></span>
                <span style="color:#94a3b8;font-size:0.75rem;">
                  Phase: <b style="color:{phase_clr};">{controller.phase}</b></span>
                <span style="color:#94a3b8;font-size:0.75rem;">
                  Allotted: <b style="color:#f1f5f9;">{total:.0f}s</b></span>
                <span style="color:#94a3b8;font-size:0.75rem;">
                  Remaining: <b style="color:{bar_clr};font-size:0.95rem;">
                  {remaining:.1f}s</b></span>
              </div>
              <div style="background:#1e293b;border-radius:6px;
                          height:10px;overflow:hidden;margin-bottom:10px;">
                <div style="width:{pct:.1f}%;height:100%;background:{bar_clr};
                            border-radius:6px;transition:width 0.4s linear;"></div>
              </div>
              <div style="font-size:0.72rem;font-weight:600;color:#60a5fa;
                          margin-bottom:6px;">📋 All Signals</div>
              {signal_rows}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Force Next always visible in AUTO
        if st.button("⏭️ Force Next Signal", use_container_width=True, type="secondary"):
            action["force_next"] = True

    # ── MANUAL panel ──────────────────────────────────────────────────────────
    else:
        st.markdown(
            "<div style='color:#94a3b8;font-size:0.78rem;margin-bottom:8px;'>"
            "🕹️ <b>MANUAL</b> — Click a signal to make it GREEN</div>",
            unsafe_allow_html=True,
        )
        btn_cols = st.columns(NUM_SIGNALS)
        for i, col in enumerate(btn_cols):
            sig       = controller.signals[i]
            is_active = (i == controller.active_idx)
            with col:
                lbl = f"{'🟢' if is_active else '🔴'} {sig.name}"
                if st.button(lbl, key=f"man_sig_{i}", use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    action["set_signal"] = i

        cnt_cols = st.columns(NUM_SIGNALS)
        for i, col in enumerate(cnt_cols):
            sig = controller.signals[i]
            with col:
                st.markdown(
                    f"<div style='text-align:center;font-size:0.7rem;"
                    f"color:#64748b;margin-top:4px;'>"
                    f"{sig.vehicle_count} veh<br>"
                    f"{traffic_badge_html(sig.traffic_level)}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        # Force Next also available in MANUAL
        if st.button("⏭️ Force Next Signal", use_container_width=True, type="secondary"):
            action["force_next"] = True

    return action


# ── CSS ───────────────────────────────────────────────────────────────────────

def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=Inter:wght@400;600&display=swap');
          html, body, [class*="css"] { font-family:'Inter',sans-serif; background-color:#0a0f1e; color:#e2e8f0; }
          h1, h2, h3 { font-family:'Rajdhani',sans-serif; letter-spacing:0.5px; }
          section[data-testid="stSidebar"] { background:#0f172a; }
          section[data-testid="stSidebar"] .stMarkdown { color:#94a3b8; }
          div[data-testid="metric-container"] {
            background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:12px;
          }
          div[data-testid="metric-container"] label { color:#64748b !important; }
          div[data-testid="metric-container"] div[data-testid="metric-value"] {
            color:#f1f5f9 !important; font-family:'Rajdhani',sans-serif; font-size:1.8rem;
          }
          .stButton > button {
            background:#1e3a5f; color:#60a5fa; border:1px solid #1d4ed8;
            border-radius:8px; font-weight:600;
          }
          .stButton > button:hover { background:#1d4ed8; color:#fff; }
          .stSlider label { color:#94a3b8; }
          hr { border-color:#1e293b; }
          img { border-radius:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
