"""Time-to-impact: when will the wheels reach the defect the camera can see?

The camera sees a defect while it is still ahead of the vehicle. The wheels reach
it later. Two earlier attempts to use the accelerometer failed because of this:

  reading the jolt at the frame's own timestamp described road the car had
  already crossed, so the sensor stream carried no information;

  taking the largest jolt over the next 3 seconds made results worse
  (precision 8.7% -> 6.4%, PR-AUC 0.274 -> 0.102), because on cobblestone the
  car shakes constantly and "rough soon" is true everywhere.

The fix is to predict a specific moment rather than scan a window.

For a forward-facing camera on flat ground, a point on the road surface at image
row y projects to a distance

    d = K / (y - y_horizon)          K = camera_height * focal_length_px

so the wheels reach it after

    tau = d / v

Then the question becomes: was there a jolt at t + tau, at the moment the
geometry predicts? Real damage produces one. Cobblestone rumble does not align
with any particular tau — it is rough at every offset — so it fails this test
while passing a plain "is it rough soon" test.

K and y_horizon are fitted from the data (see calibrate), not guessed, so the
constants can be defended.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402


@dataclass
class CameraModel:
    """Flat-ground projection for one dashcam.

    k_px_m   product of camera height and vertical focal length, in pixel-metres
    y_horizon image row of the vanishing line, in pixels
    """
    k_px_m: float
    y_horizon: float
    img_h: int = 720
    fitted: bool = False
    fit_score: float = 0.0
    n_used: int = 0

    def distance_m(self, y_bottom: float) -> Optional[float]:
        """Ground distance to the bottom edge of a box, in metres."""
        dy = y_bottom - self.y_horizon
        if dy <= 1.0:                      # at or above the horizon: unusable
            return None
        d = self.k_px_m / dy
        return float(d) if 0.5 <= d <= 120.0 else None

    def tau_s(self, y_bottom: float, speed_mps: float) -> Optional[float]:
        """Seconds until the wheels reach the point at the box's bottom edge."""
        if speed_mps is None or speed_mps < 1.0:      # too slow to predict
            return None
        d = self.distance_m(y_bottom)
        if d is None:
            return None
        tau = d / speed_mps
        return float(tau) if 0.05 <= tau <= 12.0 else None

    def to_dict(self) -> dict:
        return {"k_px_m": self.k_px_m, "y_horizon": self.y_horizon,
                "img_h": self.img_h, "fitted": self.fitted,
                "fit_score": self.fit_score, "n_used": self.n_used}

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(**{k: d[k] for k in
                      ("k_px_m", "y_horizon", "img_h", "fitted", "fit_score", "n_used")
                      if k in d})

    @classmethod
    def default(cls, img_h: int = 720) -> "CameraModel":
        """Plausible starting point for a windscreen-mounted 720p dashcam.

        Roughly 1.3 m mounting height and a 600 px vertical focal length, with
        the horizon slightly above the image centre. Used only when fitting
        fails; always reported as unfitted so it is never mistaken for measured.
        """
        return cls(k_px_m=1.3 * 600.0, y_horizon=img_h * 0.45, img_h=img_h)


class JoltSeries:
    """Surface-normalised jolt, looked up at an arbitrary time."""

    def __init__(self, t: np.ndarray, pct: np.ndarray, raw: np.ndarray,
                 surface: np.ndarray):
        order = np.argsort(t)
        self.t = t[order]
        self.pct = pct[order]
        self.raw = raw[order]
        self.surface = surface[order]

    @classmethod
    def load(cls, path: Path) -> "JoltSeries":
        z = np.load(path, allow_pickle=True)
        return cls(z["t"], z["pct"], z["raw"], z["surface"])

    def _window(self, t0: float, half: float) -> slice:
        lo = int(np.searchsorted(self.t, t0 - half, "left"))
        hi = int(np.searchsorted(self.t, t0 + half, "right"))
        return slice(lo, hi)

    def peak_pct_at(self, t0: float, half_window_s: float = 0.4) -> Optional[float]:
        """Highest surface-relative jolt percentile near t0."""
        s = self._window(t0, half_window_s)
        if s.stop <= s.start:
            return None
        return float(np.max(self.pct[s]))

    def peak_raw_at(self, t0: float, half_window_s: float = 0.4) -> Optional[float]:
        s = self._window(t0, half_window_s)
        if s.stop <= s.start:
            return None
        return float(np.max(self.raw[s]))

    def baseline_pct(self, t0: float, offsets=(-6.0, -4.5, 4.5, 6.0),
                     half_window_s: float = 0.4) -> Optional[float]:
        """Typical jolt at nearby times that are NOT the predicted impact.

        Used as a control: a genuine defect should stand above this, ordinary
        rough surface should not.
        """
        vals = [v for off in offsets
                if (v := self.peak_pct_at(t0 + off, half_window_s)) is not None]
        return float(np.mean(vals)) if vals else None


def alignment_score(cam: CameraModel, dets: list[dict], jolt: JoltSeries,
                    half_window_s: float = 0.4) -> tuple[float, int]:
    """How much the predicted impact moments stand out against nearby control times.

    Positive means jolts really do cluster where the geometry says they should.
    Near zero means the model explains nothing, and the fit should not be used.
    """
    diffs = []
    for d in dets:
        spd = d.get("speed_mps")
        t = d.get("timestamp")
        y = d.get("y_bottom")
        if None in (spd, t, y):
            continue
        tau = cam.tau_s(float(y), float(spd))
        if tau is None:
            continue
        hit = jolt.peak_pct_at(float(t) + tau, half_window_s)
        base = jolt.baseline_pct(float(t) + tau, half_window_s=half_window_s)
        if hit is None or base is None:
            continue
        diffs.append(hit - base)
    if not diffs:
        return 0.0, 0
    return float(np.mean(diffs)), len(diffs)


def calibrate(dets: list[dict], jolt: JoltSeries, img_h: int = 720,
              verbose: bool = True, top_frac: float | None = None) -> CameraModel:
    """Grid-search K and y_horizon by alignment score.

    The search deliberately reports its own weakness: if the best score is small,
    the geometry is not supported by this data and the default model is returned
    marked unfitted, so nothing downstream pretends it was measured.
    """
    # Calibration assumes the detections correspond to something real. When the
    # detector's false-positive rate is high, restricting the fit to its most
    # confident detections raises the share that are genuine and strengthens the
    # signal. Falls back to everything if that leaves too few.
    if top_frac is not None and 0 < top_frac < 1:
        scored = [d for d in dets if d.get("conf") is not None]
        if len(scored) >= 60:
            scored.sort(key=lambda d: -d["conf"])
            keep = scored[:max(50, int(len(scored) * top_frac))]
            if verbose:
                print(f"  fitting on the top {len(keep)} detections by confidence "
                      f"(of {len(dets)})")
            dets = keep

    k_grid = np.linspace(300.0, 1600.0, 27)          # ~0.5-2.7 m x 600 px
    y_grid = np.linspace(img_h * 0.35, img_h * 0.58, 24)

    best, best_k, best_y, best_n = -np.inf, None, None, 0
    for k in k_grid:
        for y in y_grid:
            cam = CameraModel(k_px_m=float(k), y_horizon=float(y), img_h=img_h)
            score, n = alignment_score(cam, dets, jolt)
            if n < 30:
                continue
            if score > best:
                best, best_k, best_y, best_n = score, float(k), float(y), n

    # A real geometric relationship showed +20 percentile points on synthetic
    # data with a known camera. Anything near zero is noise dressed as a fit.
    # The bar is deliberately high: calling noise "fitted" is worse than saying
    # the data does not support it.
    MIN_ALIGNMENT = 5.0

    if best_k is None or best < MIN_ALIGNMENT:
        cam = CameraModel.default(img_h)
        cam.fit_score = float(best) if np.isfinite(best) else 0.0
        cam.n_used = best_n
        if verbose:
            print(f"  camera fit NOT SUPPORTED "
                  f"(best alignment {cam.fit_score:+.2f} percentile points, "
                  f"need >= {MIN_ALIGNMENT:.0f}, n={best_n})")
            print("  Most likely cause: calibration assumes the detections are real,")
            print("  and on this trace almost all of them are false positives, so")
            print("  there is no impact to align to.")
            print("  Falling back to nominal parameters. The impact evidence will be")
            print("  labelled weak, and the sensor stream should be reported as")
            print("  unsupported for this trace rather than quietly used.")
        return cam

    # Coarse search uses a 0.4 s tolerance, which leaves a ridge where K and tau
    # trade off against each other. Refine locally with a tighter window so the
    # timing, which is what actually matters, is pinned down.
    for half in (0.25, 0.15):
        k_fine = np.linspace(best_k * 0.7, best_k * 1.4, 15)
        y_fine = np.linspace(best_y - 25, best_y + 25, 11)
        b2, k2, y2, n2 = -np.inf, best_k, best_y, best_n
        for k in k_fine:
            for y in y_fine:
                c = CameraModel(k_px_m=float(k), y_horizon=float(y), img_h=img_h)
                sc, n = alignment_score(c, dets, jolt, half_window_s=half)
                if n >= 30 and sc > b2:
                    b2, k2, y2, n2 = sc, float(k), float(y), n
        if np.isfinite(b2):
            best_k, best_y, best_n = k2, y2, n2

    best, best_n = alignment_score(
        CameraModel(k_px_m=best_k, y_horizon=best_y, img_h=img_h), dets, jolt)

    # A windscreen mount sits roughly 1.1-1.6 m up. A fit implying something far
    # outside that has locked onto an artefact, whatever its alignment score.
    h_est = best_k / 600.0
    plausible = 0.8 <= h_est <= 2.0
    cam = CameraModel(k_px_m=best_k, y_horizon=best_y, img_h=img_h,
                      fitted=plausible, fit_score=float(best), n_used=best_n)
    if verbose:
        print(f"  camera fit: K={best_k:.0f} px-m, horizon row {best_y:.0f}")
        print(f"    implies ~{h_est:.2f} m mounting height at a 600 px focal length")
        print(f"    alignment {best:+.1f} percentile points above control times "
              f"(n={best_n})")
        if not plausible:
            print(f"    REJECTED: {h_est:.2f} m is not a plausible dashcam height, "
                  f"so this is an artefact rather than a fit")
        else:
            print("    accepted")
    return cam
