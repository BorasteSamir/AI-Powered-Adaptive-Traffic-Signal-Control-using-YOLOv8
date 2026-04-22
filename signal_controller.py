# signal_controller.py — 4-signal corridor coordinator.
#
# Rules:
#   • On startup: ALL signals RED. S1 turns GREEN after a 1-second settle.
#   • Only ONE signal is GREEN at any time. All others are RED.
#   • AUTO: GREEN → YELLOW → ALL_RED → next GREEN. Timer is strict.
#   • Green duration locked at phase start from vehicle weight.
#   • MANUAL: operator picks which signal is GREEN; timer frozen.
#   • force_next() works in both modes.

import time
from config import (
    NUM_SIGNALS,
    THRESHOLD_LOW, THRESHOLD_MEDIUM,
    GREEN_BASE_SEC, GREEN_PER_VEHICLE, GREEN_MAX_SEC, GREEN_MIN_SEC,
    YELLOW_SEC, ALL_RED_SEC,
    COORDINATION_ENABLED, OFFSET_SEC, SKIP_EMPTY_SIGNALS,
    SignalState, SystemStats,
)

MAX_WAIT_SEC = 60.0   # force-promote a signal if it waits longer than this


def traffic_level(count: int) -> str:
    if count <= THRESHOLD_LOW:
        return "LOW"
    elif count <= THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "HIGH"


def _calc_green(weighted_score: float) -> float:
    return max(GREEN_MIN_SEC,
               min(GREEN_MAX_SEC, GREEN_BASE_SEC + weighted_score * GREEN_PER_VEHICLE))


class SignalController:

    def __init__(self) -> None:
        # All signals start RED
        self.signals: list[SignalState] = [
            SignalState(id=i + 1, name=f"S{i + 1}", state="RED")
            for i in range(NUM_SIGNALS)
        ]
        self.active_idx: int        = 0
        self.phase: str             = "STARTUP"   # brief all-red before first green
        self.phase_start: float     = time.time()
        self.green_duration: float  = GREEN_BASE_SEC
        self.stats                  = SystemStats()
        self.manual_mode: bool      = False
        self._red_since: list[float] = [time.time()] * NUM_SIGNALS

    # ── public read-only properties for UI ───────────────────────────────────

    @property
    def elapsed(self) -> float:
        return max(0.0, time.time() - self.phase_start)

    @property
    def remaining(self) -> float:
        """Seconds left in the current GREEN phase (0 during YELLOW/ALL_RED)."""
        if self.phase == "GREEN":
            return max(0.0, self.green_duration - self.elapsed)
        return 0.0

    @property
    def yellow_remaining(self) -> float:
        """Seconds left in YELLOW phase."""
        if self.phase == "YELLOW":
            return max(0.0, YELLOW_SEC - self.elapsed)
        return 0.0

    @property
    def progress(self) -> float:
        """0.0→1.0 drain fraction for the green bar (1.0 = full, 0.0 = expired)."""
        if self.phase == "GREEN" and self.green_duration > 0:
            return max(0.0, 1.0 - self.elapsed / self.green_duration)
        return 0.0

    # ── mode switching ────────────────────────────────────────────────────────

    def set_auto(self) -> None:
        self.manual_mode    = False
        self.phase          = "GREEN"
        self.phase_start    = time.time()
        sig                 = self.signals[self.active_idx]
        self.green_duration = _calc_green(sig.weighted_score)
        sig.green_time      = self.green_duration

    def set_manual(self) -> None:
        self.manual_mode = True

    # ── manual: pick a specific signal ───────────────────────────────────────

    def set_signal(self, idx: int) -> None:
        now = time.time()
        for i, sig in enumerate(self.signals):
            sig.state   = "GREEN" if i == idx else "RED"
            sig.skipped = False
            if i != idx:
                self._red_since[i] = now
        self.active_idx          = idx
        self.phase               = "GREEN"
        self.phase_start         = now
        self._red_since[idx]     = 0.0
        self.green_duration      = _calc_green(self.signals[idx].weighted_score)
        self.signals[idx].green_time = self.green_duration
        self._sync_stats()

    # ── force advance (manual intervention, works in both modes) ─────────────

    def force_next(self) -> None:
        now = time.time()
        self.signals[self.active_idx].state = "RED"
        self._red_since[self.active_idx]    = now
        self.active_idx = self._next_idx(self.active_idx)
        self._start_green(now)
        self._sync_stats()

    # ── per-frame update ──────────────────────────────────────────────────────

    def update_signals(self, zone_detections: list[list[dict]]) -> None:
        now     = time.time()
        elapsed = now - self.phase_start

        # Refresh counts for every zone
        for i, dets in enumerate(zone_detections):
            sig                = self.signals[i]
            sig.detections     = dets
            sig.vehicle_count  = len(dets)
            sig.weighted_score = sum(d["weight"] for d in dets)
            sig.traffic_level  = traffic_level(sig.vehicle_count)
            sig.wait_time      = (now - self._red_since[i]
                                  if sig.state == "RED" and self._red_since[i] > 0
                                  else 0.0)

        self.signals[self.active_idx].green_time = self.green_duration

        if not self.manual_mode:
            # ── STARTUP: brief all-red before first green ─────────────────
            if self.phase == "STARTUP":
                if elapsed >= 1.0:
                    self._start_green(now)

            # ── GREEN: wait full allotted time, then go YELLOW ────────────
            elif self.phase == "GREEN":
                if elapsed >= self.green_duration:
                    self.signals[self.active_idx].state = "YELLOW"
                    self.phase       = "YELLOW"
                    self.phase_start = now

            # ── YELLOW: wait YELLOW_SEC, then ALL_RED ─────────────────────
            elif self.phase == "YELLOW":
                if elapsed >= YELLOW_SEC:
                    self.signals[self.active_idx].state = "ALL_RED"
                    self.phase       = "ALL_RED"
                    self.phase_start = now
                    self._red_since[self.active_idx] = now

            # ── ALL_RED: safety clearance, then next GREEN ────────────────
            elif self.phase == "ALL_RED":
                if elapsed >= ALL_RED_SEC:
                    self.signals[self.active_idx].state = "RED"
                    self.active_idx = self._next_idx(self.active_idx)
                    if self.active_idx == 0:
                        self.stats.cycle_count += 1
                    offset = OFFSET_SEC if COORDINATION_ENABLED else 0.0
                    self._start_green(now - offset)

        self._sync_stats()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _next_idx(self, current: int) -> int:
        """Round-robin with starvation prevention and optional empty-skip."""
        now = time.time()
        # Starvation check first
        for off in range(1, NUM_SIGNALS):
            c = (current + off) % NUM_SIGNALS
            if self._red_since[c] > 0 and (now - self._red_since[c]) >= MAX_WAIT_SEC:
                self.signals[c].skipped = False
                return c
        # Normal sequential, skip empty if enabled
        for off in range(1, NUM_SIGNALS):
            c   = (current + off) % NUM_SIGNALS
            sig = self.signals[c]
            if SKIP_EMPTY_SIGNALS and sig.vehicle_count == 0:
                sig.skipped = True
                self.stats.skipped_count += 1
                continue
            sig.skipped = False
            return c
        # All empty — just advance
        nxt = (current + 1) % NUM_SIGNALS
        self.signals[nxt].skipped = False
        return nxt

    def _start_green(self, now: float) -> None:
        sig                      = self.signals[self.active_idx]
        sig.state                = "GREEN"
        sig.skipped              = False
        self.phase               = "GREEN"
        self.phase_start         = now
        self.green_duration      = _calc_green(sig.weighted_score)
        sig.green_time           = self.green_duration
        self._red_since[self.active_idx] = 0.0
        self.stats.green_wave_active = COORDINATION_ENABLED

    def _sync_stats(self) -> None:
        self.stats.active_signal  = self.signals[self.active_idx].id
        self.stats.active_state   = self.signals[self.active_idx].state
        self.stats.total_vehicles = sum(s.vehicle_count for s in self.signals)
        waiting = [s.wait_time for s in self.signals
                   if s.state == "RED" and s.wait_time > 0]
        self.stats.avg_wait_time = sum(waiting) / len(waiting) if waiting else 0.0
