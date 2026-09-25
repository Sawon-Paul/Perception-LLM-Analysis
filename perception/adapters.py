"""Turn one verified detection into a blockchain report and into training data.

Plan section 5.3. The LLM and its prompt are unchanged; these run after its
JSON is parsed and send the result two ways.

Adapter A builds the on-chain report. Nothing identifying goes on the chain:
the crop, the LLM reasoning and the box are bundled into a file, and only the
SHA-256 of that bundle is reported. Location is a geohash cell, not a raw GPS
fix, so a stored report cannot be turned back into a vehicle's exact track.

Adapter B writes the YOLO labels for the next FL round, and a manifest row
saying which fault each sample came from. Without the manifest, sample
unlearning cannot find what to delete when a fault turns out to be fake.
"""
from __future__ import annotations

import csv
import hashlib
import json
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

FAULT_TYPE_IDS = {"pothole": 0, "alligator cracking": 1, "lateral cracking": 2,
                  "longitudinal cracking": 3, "missing sign": 4}

_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(lat: float, lon: float, precision: int = 8) -> str:
    """Geohash without a third-party package, so the car has one less dependency.

    Precision 8 is about 38 x 19 m — roughly GPS error, and coarse enough that
    a published report does not pin a vehicle to a lane.
    """
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    out, bit, ch, even = [], 0, 0, True
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon > mid:
                ch = (ch << 1) | 1
                lon_lo = mid
            else:
                ch <<= 1
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat > mid:
                ch = (ch << 1) | 1
                lat_lo = mid
            else:
                ch <<= 1
                lat_hi = mid
        even = not even
        bit += 1
        if bit == 5:
            out.append(_B32[ch])
            bit, ch = 0, 0
    return "".join(out)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ Adapter A

@dataclass
class ReportPayload:
    fault_type_id: int
    geohash8: str
    severity: int
    yolo_conf: float
    llm_conf: float
    evidence_hash: str
    observed_at: int          # unix seconds, kept inside the payload so an
                              # offline-queued report keeps its real time

    def as_tx_args(self) -> tuple:
        """Positional arguments for FaultLifecycle.report(...)."""
        return (self.fault_type_id, self.geohash8, self.severity,
                int(round(self.yolo_conf * 1000)),
                int(round(self.llm_conf * 1000)),
                bytes.fromhex(self.evidence_hash))


def build_evidence_bundle(crop_path: str | Path, llm_json: dict,
                          yolo_box: dict, out_dir: str | Path) -> tuple[Path, str]:
    """Bundle crop + LLM JSON + box into one file named by its own hash.

    Written to a temporary name first and renamed to its digest, so the store
    is content-addressed: the same evidence always lands on the same name and
    the hash on-chain always matches the file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".building_{int(time.time() * 1000)}.tar"

    # No timestamp in the metadata and no mtimes in the archive: the bundle has
    # to be content-addressed, so the same evidence must always hash the same.
    # A clock reading inside it made every rebuild produce a new hash.
    meta = {"llm": llm_json, "yolo": yolo_box, "schema": 1}
    meta_path = out_dir / f".meta_{int(time.time() * 1000)}.json"
    meta_path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")

    try:
        def deterministic(ti: tarfile.TarInfo) -> tarfile.TarInfo:
            ti.mtime = 0
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.mode = 0o644
            return ti

        with tarfile.open(tmp, "w", format=tarfile.GNU_FORMAT) as tar:
            tar.add(crop_path, arcname="crop.jpg", filter=deterministic)
            tar.add(meta_path, arcname="evidence.json", filter=deterministic)
        digest = sha256_file(tmp)
        final = out_dir / f"{digest}.tar"
        if final.exists():
            tmp.unlink()
        else:
            tmp.rename(final)
        return final, digest
    finally:
        meta_path.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)


def adapter_a(event: dict, llm: dict, evidence_dir: str | Path) -> ReportPayload | None:
    """One verified detection -> the payload for FaultLifecycle.report().

    Returns None when there is no crop to evidence, because a report whose
    evidence hash points at nothing cannot be audited later.
    """
    crop = event.get("stream1_image", {}).get("crop_path")
    if not crop or not Path(crop).exists():
        return None

    s5 = event.get("stream5_model", {})
    box = {"bbox_xyxy": event["stream1_image"].get("bbox_xyxy"),
           "class_name": s5.get("class_name"), "conf": s5.get("conf"),
           "detector": s5.get("detector")}
    _, digest = build_evidence_bundle(crop, llm, box, evidence_dir)

    name = str(llm.get("fault_type", s5.get("class_name", ""))).lower().strip()
    type_id = FAULT_TYPE_IDS.get(name)
    if type_id is None:
        type_id = FAULT_TYPE_IDS.get(str(s5.get("class_name", "")).lower(), 0)

    lat, lon = event.get("lat"), event.get("lon")
    if lat is None or lon is None:
        return None

    return ReportPayload(
        fault_type_id=type_id,
        geohash8=geohash_encode(float(lat), float(lon), 8),
        severity=int(llm.get("severity", 1)),
        yolo_conf=float(s5.get("conf") or 0.0),
        llm_conf=float(llm.get("confidence") or 0.0),
        evidence_hash=digest,
        observed_at=int(float(event.get("timestamp") or time.time())),
    )


# ------------------------------------------------------------------ Adapter B

def to_yolo_line(bbox_xyxy, img_w: int, img_h: int, class_id: int) -> str | None:
    x1, y1, x2, y2 = bbox_xyxy
    x1, x2 = max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))
    y1, y2 = max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))
    bw, bh = abs(x2 - x1) / img_w, abs(y2 - y1) / img_h
    if bw <= 0 or bh <= 0:
        return None
    return (f"{class_id} {((x1 + x2) / 2) / img_w:.6f} "
            f"{((y1 + y2) / 2) / img_h:.6f} {bw:.6f} {bh:.6f}")


def adapter_b(events: list[dict], labels_dir: str | Path,
              manifest_path: str | Path, round_created: int,
              fault_ids: dict[str, str] | None = None) -> dict[str, int]:
    """Verified detections -> YOLO labels + the manifest unlearning needs.

    A verified box is written. A rejected one is not, and a frame with no
    verified boxes gets an empty label file — that is how YOLO is taught the
    surface is fine, and it is what cut detections from 366 to 27.

    Every sample is recorded against the fault it belongs to, so when a fault
    is later rejected on-chain the exact frames can be found and removed.
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    fault_ids = fault_ids or {}

    by_frame: dict[str, list[dict]] = {}
    for e in events:
        fp = e.get("stream1_image", {}).get("frame_path")
        if fp:
            by_frame.setdefault(fp, []).append(e)

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not manifest_path.exists()
    n_boxes = n_bg = 0

    with manifest_path.open("a", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        if new_file:
            w.writerow(["frame_id", "round_created", "fault_id",
                        "label_source", "n_boxes", "written_at"])

        for frame_path, evs in by_frame.items():
            stem = Path(frame_path).stem
            lines, faults = [], set()
            for e in evs:
                p2 = e.get("phase2") or {}
                if not p2.get("verified"):
                    continue          # rejected: no box, this region is background
                s5 = e.get("stream5_model", {})
                cid = s5.get("class_id")
                if cid is None:
                    cid = FAULT_TYPE_IDS.get(
                        str(p2.get("fault_type", "")).lower(), 0)
                line = to_yolo_line(e["stream1_image"]["bbox_xyxy"],
                                    e.get("img_w") or 1280,
                                    e.get("img_h") or 720, int(cid))
                if line:
                    lines.append(line)
                    if (fid := fault_ids.get(e["detection_id"])):
                        faults.add(fid)

            (labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            if lines:
                n_boxes += len(lines)
            else:
                n_bg += 1
            w.writerow([stem, round_created, "|".join(sorted(faults)),
                        "llm_teacher", len(lines), int(time.time())])

    return {"frames": len(by_frame), "boxes": n_boxes, "background": n_bg,
            "manifest": str(manifest_path)}


def samples_for_fault(manifest_path: str | Path, fault_id: str) -> list[str]:
    """Which frames came from one fault — the lookup sample unlearning runs."""
    out = []
    p = Path(manifest_path)
    if not p.exists():
        return out
    for r in csv.DictReader(p.open(encoding="utf-8")):
        if fault_id in (r.get("fault_id") or "").split("|"):
            out.append(r["frame_id"])
    return out
