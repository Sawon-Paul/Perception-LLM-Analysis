"""YOLO wrappers for federated learning — plan section 5.2.

Four things the FL loop needs from the detector:

  load_global()          fetch the round's global weights, verify the hash first
  get_trainable_state()  the tensors a car sends to the server
  set_trainable_state()  load a global update without touching the backbone
  train_local()          one car's local round
  evaluate()             precision, recall, F1, mAP50 on a labelled folder

Only layers 10 and above travel. Layers 0-9 are the backbone, frozen and
identical in every car, so sending them would waste bandwidth and let a
malicious client rewrite the shared feature extractor.

BatchNorm running statistics ARE included. They are buffers, not parameters, so
a naive `named_parameters()` filter silently drops them and every car quietly
diverges on normalisation while the weights look synchronised.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

# Layer 10 onward: neck + head. Matches freeze=10 in the training call.
FIRST_TRAINABLE_LAYER = 10
_LAYER_RE = re.compile(r"(?:^|\.)model\.(\d+)\.")


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _layer_index(key: str) -> int | None:
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def is_trainable_key(key: str, first: int = FIRST_TRAINABLE_LAYER) -> bool:
    idx = _layer_index(key)
    return idx is not None and idx >= first


# ----------------------------------------------------------------- state dict

def get_trainable_state(model, first: int = FIRST_TRAINABLE_LAYER) -> dict[str, torch.Tensor]:
    """The part of the model a car sends to the FL server.

    Includes BatchNorm running_mean, running_var and num_batches_tracked for
    the unfrozen layers — they are buffers, and leaving them out desynchronises
    normalisation while the weights appear to match.
    """
    inner = model.model if hasattr(model, "model") else model
    sd = inner.state_dict()
    return {k: v.detach().cpu().clone() for k, v in sd.items()
            if is_trainable_key(k, first)}


def set_trainable_state(model, state: dict[str, torch.Tensor],
                        first: int = FIRST_TRAINABLE_LAYER,
                        strict_shapes: bool = True) -> int:
    """Load a trainable-layer state dict back. The backbone is untouched.

    Refuses keys outside the trainable range, so a malformed or hostile update
    cannot quietly rewrite the shared backbone.
    """
    inner = model.model if hasattr(model, "model") else model
    own = inner.state_dict()
    loaded = 0
    for k, v in state.items():
        if not is_trainable_key(k, first):
            raise ValueError(f"{k} is outside the trainable range "
                             f"(layer >= {first}); refusing to load it")
        if k not in own:
            raise KeyError(f"{k} is not in this model")
        if strict_shapes and own[k].shape != v.shape:
            raise ValueError(f"{k}: expected {tuple(own[k].shape)}, "
                             f"got {tuple(v.shape)}")
        own[k] = v.to(own[k].dtype).to(own[k].device)
        loaded += 1
    inner.load_state_dict(own, strict=True)
    return loaded


def state_summary(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    n_params = sum(v.numel() for v in state.values())
    bn = sum(1 for k in state if k.endswith(("running_mean", "running_var",
                                             "num_batches_tracked")))
    layers = sorted({_layer_index(k) for k in state} - {None})
    return {"tensors": len(state), "parameters": n_params,
            "bn_buffers": bn, "layers": layers,
            "megabytes": round(n_params * 4 / 1024 ** 2, 2)}


def state_distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    """L2 distance between two updates — the FL divergence plot in 11.6."""
    total = 0.0
    for k in a:
        if k in b and a[k].dtype.is_floating_point:
            total += float(torch.sum((a[k].float() - b[k].float()) ** 2))
    return total ** 0.5


def save_update(state: dict[str, torch.Tensor], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return sha256(path)


def load_update(path: str | Path, expect_hash: str | None = None) -> dict[str, torch.Tensor]:
    """Load an update, refusing it if the hash does not match the chain.

    This is the defence against the model-file-swap attack in 13.1: the server
    may host the file, but the hash is on-chain, and a mismatch means refuse
    and log rather than load.
    """
    got = sha256(path)
    if expect_hash is not None and got != expect_hash:
        raise ValueError(f"hash mismatch for {path}\n  on-chain {expect_hash}\n"
                         f"  on-disk  {got}\nrefusing to load")
    return torch.load(path, map_location="cpu", weights_only=True)


# ------------------------------------------------------------------ training

def load_global(weights_path: str | Path, expect_hash: str | None = None):
    """Open a global checkpoint after checking its hash against the chain."""
    from ultralytics import YOLO
    got = sha256(weights_path)
    if expect_hash is not None and got != expect_hash:
        raise ValueError(f"global model hash mismatch: on-chain {expect_hash}, "
                         f"on-disk {got}. Refusing to load.")
    return YOLO(str(weights_path))


def train_local(weights: str | Path, data_yaml: str | Path, out_dir: str | Path,
                epochs: int = 2, lr: float = 0.0005, batch: int = 16,
                imgsz: int = 640, freeze: int = FIRST_TRAINABLE_LAYER,
                device: int | str = 0, name: str = "local") -> dict[str, Any]:
    """One car's local training round.

    Same settings that produced the fine-tuned detector: backbone frozen, low
    learning rate, augmentation kept mild so the model learns this road surface
    rather than invented variety.
    """
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    results = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        lr0=lr, lrf=0.1, freeze=freeze, optimizer="AdamW",
        warmup_epochs=1.0, mosaic=0.0, mixup=0.0, degrees=0.0, shear=0.0,
        perspective=0.0, fliplr=0.5, hsv_v=0.3,
        device=device, project=str(out_dir), name=name, exist_ok=True,
        val=True, plots=False, verbose=False,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    return {"weights": str(best), "save_dir": str(results.save_dir),
            "epochs": epochs, "lr": lr}


def _sibling_labels_dir(images_dir: Path) -> Path:
    """Map .../images/<split> to .../labels/<split>, wherever "images" sits."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels" / images_dir.name


def evaluate(weights: str | Path, images_dir: str | Path,
             labels_dir: str | Path | None = None,
             class_names: dict[int, str] | None = None,
             imgsz: int = 640, device: int | str = 0) -> dict[str, float]:
    """Precision, recall, F1 and mAP50 on a labelled folder.

    Builds a throwaway data.yaml because Ultralytics validates against a
    dataset spec rather than a bare folder.
    """
    from ultralytics import YOLO

    images_dir = Path(images_dir).resolve()
    if labels_dir:
        labels_dir = Path(labels_dir).resolve()
    else:
        # YOLO layout is <root>/images/<split> and <root>/labels/<split>, so the
        # "images" component two levels up is what becomes "labels". Deriving it
        # from images_dir.parent instead gave <root>/images/labels/<split>,
        # which exists nowhere, and every metric came back 0.0 with no error.
        labels_dir = _sibling_labels_dir(images_dir)

    model = YOLO(str(weights))
    names = class_names or model.names

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"labels folder not found: {labels_dir}\n"
            f"Pass labels_dir= explicitly if your layout differs.")

    tmp = Path(tempfile.mkdtemp())
    root = tmp / "ds"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)

    n_images = n_label_files = n_boxes = 0
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        shutil.copy(p, root / "images" / "val" / p.name)
        n_images += 1
        lab = labels_dir / f"{p.stem}.txt"
        body = lab.read_text(encoding="utf-8") if lab.exists() else ""
        if lab.exists():
            n_label_files += 1
            n_boxes += sum(1 for ln in body.splitlines() if ln.strip())
        (root / "labels" / "val" / f"{p.stem}.txt").write_text(body, encoding="utf-8")

    # Reporting 0.0 across the board because the labels were not found is worse
    # than failing: in an FL loop it looks like the model collapsed.
    if n_label_files == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise FileNotFoundError(
            f"{n_images} images in {images_dir}, but no .txt labels in "
            f"{labels_dir}. Metrics would all be 0.0 for the wrong reason.")
    if n_boxes == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(
            f"{n_label_files} label files found but every one is empty "
            f"(all background). There is nothing to score against.")

    lines = [f"path: {root.as_posix()}", "train: images/val", "val: images/val",
             f"nc: {len(names)}", "names:"]
    for i in sorted(names):
        lines.append(f"  {i}: {names[i]}")
    yaml_path = tmp / "data.yaml"
    yaml_path.write_text("\n".join(lines), encoding="utf-8")

    try:
        m = model.val(data=str(yaml_path), imgsz=imgsz, device=device,
                      verbose=False, plots=False)
        p, r = float(m.box.mp), float(m.box.mr)
        return {"precision": round(p, 4), "recall": round(r, 4),
                "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                "map50": round(float(m.box.map50), 4),
                "map50_95": round(float(m.box.map), 4),
                "n_images": n_images, "n_boxes": n_boxes}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
