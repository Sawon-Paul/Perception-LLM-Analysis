import json
p = r"C:\Sawon\LLM\data\events\events_PVS2_left_p2_standalone_7b.jsonl"
r = [json.loads(l) for l in open(p, encoding="utf-8")]
v = [e["phase2"] for e in r if (e.get("phase2") or {}).get("verified")]
print(len(v), "accepted\n")
for x in v:
    print(f"[{x['fault_type']}] sev{x['severity']} conf{x['confidence']}")
    print(f"   saw : {x['observed']}")
    print(f"   why : {x['reason']}\n")