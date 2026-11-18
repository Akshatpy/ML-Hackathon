"""Track cleaning, gap segmentation and per-fix kinematics.

Fixes are never dropped: the API must return a road segment for every
observation, so bad fixes are flagged and handled downstream instead.
"""
import numpy as np


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Map radians into (-pi, pi]."""
    return np.angle(np.exp(1j * a))


def segment_ids(t: np.ndarray, gap_s: float) -> np.ndarray:
    """New segment wherever the time step exceeds gap_s. The HMM restarts there
    because a 14 h hole carries no information about connectivity."""
    return np.concatenate([[0], np.cumsum(np.diff(t) > gap_s)])


def _neighbour_index(n: int, seg: np.ndarray, offset: int) -> np.ndarray:
    """Index of fix i+offset, clamped to stay inside fix i's segment."""
    starts = np.searchsorted(seg, seg, side="left")
    ends = np.searchsorted(seg, seg, side="right") - 1
    return np.clip(np.arange(n) + offset, starts, ends)


def _window_heading(xy, t, seg, half_window):
    n = len(t)
    prev_i = _neighbour_index(n, seg, -half_window)
    next_i = _neighbour_index(n, seg, half_window)
    disp = xy[next_i] - xy[prev_i]
    return np.arctan2(disp[:, 0], disp[:, 1]), np.linalg.norm(disp, axis=1), t[next_i] - t[prev_i]


def _carry_within_segment(values: np.ndarray, known: np.ndarray, seg: np.ndarray):
    """Fill unknown entries from the closest earlier known fix in the same segment,
    else the closest later one. Returns (filled values, filled mask)."""
    n = len(values)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(known, idx, -1))
    nxt = np.minimum.accumulate(np.where(known, idx, n)[::-1])[::-1]
    use_last = (last >= 0) & (seg[np.maximum(last, 0)] == seg)
    use_next = ~use_last & (nxt < n) & (seg[np.minimum(nxt, n - 1)] == seg)
    src = np.where(use_last, last, np.where(use_next, nxt, idx))
    filled = known | use_last | use_next
    return values[src], filled


def kinematics(xy: np.ndarray, t: np.ndarray, seg: np.ndarray, half_window: int,
               min_disp: float, speed_clip: float, mode: str = "adaptive",
               adaptive_windows=(2, 4, 8, 15), adaptive_min_disp: float = 8.0) -> dict:
    n = len(t)
    heading, span, dt_span = _window_heading(xy, t, seg, half_window)
    heading_ok = span >= min_disp                           # stationary jitter gives random bearings
    speed = np.divide(span, dt_span, out=np.zeros(n), where=dt_span > 0)

    carried = np.zeros(n, bool)
    if mode == "adaptive":
        # Slow traffic: widen the window until the displacement is long enough that
        # GPS noise (~5 m) can't flip the bearing; at a standstill, reuse the heading
        # the vehicle had when it stopped. Error analysis: 37% of heading-rule errors
        # were at speed < 2 m/s.
        heading = np.full(n, np.nan)
        for w in adaptive_windows:
            h_w, span_w, _ = _window_heading(xy, t, seg, w)
            take = np.isnan(heading) & (span_w >= adaptive_min_disp)
            heading[take] = h_w[take]
        known = ~np.isnan(heading)
        heading, filled = _carry_within_segment(np.nan_to_num(heading), known, seg)
        carried = filled & ~known
        heading_ok = filled

    step_prev = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    dt_prev = np.r_[0.0, np.diff(t)]
    new_seg = np.r_[True, seg[1:] != seg[:-1]]
    step_prev[new_seg] = 0.0
    dt_prev[new_seg] = 0.0

    # turning: change between the incoming and outgoing headings
    v_in = xy - xy[_neighbour_index(n, seg, -1)]
    v_out = xy[_neighbour_index(n, seg, 1)] - xy
    h_in = np.arctan2(v_in[:, 0], v_in[:, 1])
    h_out = np.arctan2(v_out[:, 0], v_out[:, 1])
    turn = np.abs(wrap_angle(h_out - h_in))

    return {
        "heading": heading,
        "heading_ok": heading_ok,
        "heading_carried": carried,
        "speed": np.clip(speed, 0, speed_clip),
        "step_prev": step_prev,
        "dt_prev": dt_prev,
        "turn": np.where(heading_ok, turn, 0.0),
    }


def prepare_trip(trip, cfg: dict, labels: dict | None = None) -> dict:
    """Everything per-fix that features and the decoder need."""
    pp, ft = cfg["preprocess"], cfg["features"]
    seg = segment_ids(trip.t, pp["segment_gap_s"])
    kin = kinematics(trip.track_xy, trip.t, seg, ft["heading_half_window"],
                     ft["min_heading_disp_m"], pp["speed_clip_mps"],
                     mode=ft.get("heading_mode", "fixed"),
                     adaptive_windows=tuple(ft.get("adaptive_windows", (2, 4, 8, 15))),
                     adaptive_min_disp=ft.get("adaptive_min_disp_m", 8.0))
    out = {"segment": seg, **kin}
    if labels is not None:
        out["label_arc"] = labels["arc_id"]
        out["label_dist"] = labels["dist"]
        valid = labels["dist"] <= pp["label_valid_m"]
        # Trip-level guard: when the typical fix is far from its reference arc, the route
        # itself is wrong (trips 33 and 54: medians 122 m and 140 m; every other trip is
        # under 12 m), and fixes that happen to pass within 50 m are matched to the wrong
        # part of it. Model check: trip 33's "valid" fixes scored 51% top-1, others 72-99%.
        if np.median(labels["dist"]) > pp["label_valid_m"]:
            valid[:] = False
        out["valid"] = valid
    return out
