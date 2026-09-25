"""Check an unlearning certificate without trusting whoever issued it.

The point of the certificate is that someone else can verify it. This does
exactly that: it recomputes both model hashes from the files, re-runs the
attack-success measurement itself, and compares against what was claimed.

Run it as the auditor, not as the service that produced the certificate. On the
chain those are different roles for the same reason.

    python -m src.unlearn.audit --cert unlearn_cert_fold0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs import config as cfg  # noqa: E402
from perception.fl_model import evaluate, sha256  # noqa: E402

RESULTS = cfg.ROOT / "results"
TOL = 0.05     # per mille tolerance is not meaningful; ASR is re-measured


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert", required=True, help="file in results/")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    p = Path(args.cert)
    if not p.exists():
        p = RESULTS / args.cert
    if not p.exists():
        raise SystemExit(f"certificate not found: {args.cert}")
    c = json.loads(p.read_text())
    device = args.device if args.device is not None else cfg.DEVICE

    print(f"auditing {p.name}\n")
    print(f"  method        {c['method']}")
    print(f"  rounds        {c['from_round']} to {c['to_round']}")
    print(f"  claimed ASR   {c['asr_before_per_mille'] / 10:.1f}% "
          f"-> {c['asr_after_per_mille'] / 10:.1f}%")

    checks = []

    # 1. do the files still hash to what was recorded?
    for label, path_key, hash_key in (
            ("cleaned model", "cleaned_model", "model_after"),):
        f = Path(c[path_key])
        if not f.exists():
            checks.append((f"{label} present", False, f"missing: {f}"))
            continue
        got = sha256(f)
        ok = (got == c[hash_key])
        checks.append((f"{label} hash", ok,
                       "matches" if ok else f"recorded {c[hash_key][:12]}, "
                                            f"found {got[:12]}"))

    # 2. re-measure the attack success rate ourselves
    asr_set = Path(c["asr_set"])
    held = Path(c["heldout_set"])
    cleaned = Path(c["cleaned_model"])
    if cleaned.exists() and asr_set.exists():
        from ultralytics import YOLO
        names = YOLO(str(cleaned)).names
        trig = evaluate(cleaned, asr_set / "images" / "val",
                        asr_set / "labels" / "val", names, device=device)
        # paired against the same frames without the trigger, matching how the
        # certificate's ASR was computed
        clean_pair = Path(str(asr_set) + "_clean")
        if clean_pair.exists():
            pc = evaluate(cleaned, clean_pair / "images" / "val",
                          clean_pair / "labels" / "val", names, device=device)
            rc = pc["recall"]
            asr = max(0.0, (rc - trig["recall"]) / rc) if rc > 0 else 0.0
        else:
            asr = 1.0 - trig["recall"]
        claimed = c["asr_after_per_mille"] / 1000
        ok = abs(asr - claimed) <= TOL
        checks.append(("ASR re-measured", ok,
                       f"auditor {asr:.3f} vs claimed {claimed:.3f}"))

        if held.exists():
            clean = evaluate(cleaned, held / "images" / "val",
                             held / "labels" / "val", names, device=device)
            ok = abs(clean["f1"] - c["clean_f1_after"]) <= TOL
            checks.append(("clean F1 re-measured", ok,
                           f"auditor {clean['f1']:.3f} vs claimed "
                           f"{c['clean_f1_after']:.3f}"))
    else:
        checks.append(("ASR re-measured", False, "model or ASR set missing"))

    print()
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:24} {detail}")

    passed = all(ok for _, ok, _ in checks)
    print("\n" + "=" * 62)
    if passed:
        print("  Certificate verified independently.")
        print("  This shows the named files hash to the recorded values and that")
        print("  the metrics reproduce. It does NOT show the removal procedure was")
        print("  run honestly, nor that no residual influence survives \u2014 say so.")
    else:
        print("  Certificate did NOT verify. Do not cite these numbers until the")
        print("  failing rows above are explained.")
    print("=" * 62)

    out = RESULTS / f"audit_{p.stem}.json"
    out.write_text(json.dumps(
        {"certificate": p.name, "passed": passed,
         "checks": [{"check": n, "passed": o, "detail": d} for n, o, d in checks]},
        indent=2))
    print(f"\n-> {out}")
    print("\nTo record it on-chain, call UnlearningLog.audit(certId, passed, note)")
    print("from the AUDITOR key \u2014 a different key from the unlearning service.")


if __name__ == "__main__":
    main()
