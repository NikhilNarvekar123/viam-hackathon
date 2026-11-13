import numpy as np


def print_summary(records: list[dict], pass_bin: str):
    known = [r for r in records if r["ground_truth"] is not None]
    false_accepts = sum(r["ground_truth"] == "defect" and r["final_bin"] == pass_bin
                        for r in known)
    false_rejects = sum(r["ground_truth"] == "good" and r["final_bin"] != pass_bin
                        for r in known)
    accuracy = ((len(known) - false_accepts - false_rejects) / len(known)
                if known else None)
    print(f"Items: {len(records)} (labeled: {len(known)})")
    print("Sorting accuracy: " + (f"{accuracy:.1%}" if accuracy is not None else "N/A"))
    print(f"False accepts: {false_accepts}; false rejects: {false_rejects}")
    print(f"Invalid VLM outputs: {sum(r['vlm_valid'] is False for r in records)}")
    print("Latency ms (decide includes zeros for skipped items):")
    for stage in ("capture", "detect", "decide", "sort"):
        values = [r["latency_ms"][stage] for r in records]
        print(f"  {stage:7s} mean={np.mean(values):.3f} p95={np.percentile(values, 95):.3f}")
