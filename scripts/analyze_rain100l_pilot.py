"""Summarize the paired baseline/DCTTA Rain100L Pilot and flag regressions."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_json(path):
    return json.loads(path.read_text())


def panel(pilot_dir, sample, destination):
    sources = [("Rainy Query", pilot_dir / "query" / sample),
               ("Clean GT", pilot_dir / "gt" / sample),
               ("AWRaCLe", pilot_dir / "baseline" / sample),
               ("AWRaCLe + DCTTA", pilot_dir / "dctta_8" / sample)]
    images = [Image.open(path).convert("RGB") for _, path in sources]
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("Comparison panel images have different dimensions")
    canvas = Image.new("RGB", (2 * width, 2 * (height + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, ((label, _), image) in enumerate(zip(sources, images)):
        left, top = (index % 2) * width, (index // 2) * (height + 28)
        draw.text((left + 8, top + 7), label, fill="black")
        canvas.paste(image, (left, top + 28))
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    args = parser.parse_args()
    pilot_dir = args.pilot_dir.resolve()
    selection = read_json(pilot_dir / "selection_audit.json")
    baseline = read_json(pilot_dir / "baseline" / "summary.json")
    adapted = read_json(pilot_dir / "dctta_8" / "metrics.json")
    adapted_summary = read_json(pilot_dir / "dctta_8" / "summary.json")
    if adapted["support_protocol"] != "seen_pool" or adapted["gt_used_for_adaptation"]:
        raise ValueError("Pilot adaptation protocol is not GT-free Seen Support Pool")
    if adapted["support_manifest_sha256"] is None or not selection["support_path_and_sha256_disjoint_from_all_test_images"]:
        raise ValueError("Support provenance has not been established")
    expected = {pair["id"] + ".png" for pair in selection["query_gt_pairs"]}
    baseline_rows = {row["sample"]: row for row in baseline["samples"]}
    adapted_rows = {row["sample"]: row for row in adapted["samples"]}
    if expected != set(baseline_rows) or expected != set(adapted_rows):
        raise ValueError("Baseline/DCTTA outputs do not cover the selected Pilot queries")

    rows = []
    for sample in sorted(expected, key=lambda name: int(Path(name).stem)):
        before, after = baseline_rows[sample], adapted_rows[sample]
        values = [before["psnr"], before["ssim"], after["psnr"], after["ssim"]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite image metric for {sample}")
        steps = after["adaptation"]
        if len(steps) != adapted["iterations"] or any(not math.isfinite(step["loss"]) or step["updated_elements"] <= 0 for step in steps):
            raise ValueError(f"Adaptation failure for {sample}")
        baseline_image = np.asarray(Image.open(pilot_dir / "baseline" / sample).convert("RGB"), dtype=np.int16)
        adapted_image = np.asarray(Image.open(pilot_dir / "dctta_8" / sample).convert("RGB"), dtype=np.int16)
        if baseline_image.shape != adapted_image.shape:
            raise ValueError(f"Output shape mismatch for {sample}")
        difference = np.abs(baseline_image - adapted_image)
        row = {"sample": sample, "width": baseline_image.shape[1], "height": baseline_image.shape[0],
               "baseline_psnr": before["psnr"], "dctta_psnr": after["psnr"],
               "delta_psnr": after["psnr"] - before["psnr"],
               "baseline_ssim": before["ssim"], "dctta_ssim": after["ssim"],
               "delta_ssim": after["ssim"] - before["ssim"],
               "baseline_evaluation_seconds": before["evaluation_seconds"],
               "dctta_adaptation_seconds": after["adaptation_seconds"],
               "dctta_evaluation_seconds": after["evaluation_seconds"],
               "output_mean_absolute_difference_255": float(difference.mean()),
               "updated_elements": sum(step["updated_elements"] for step in steps),
               "material_regression": (after["psnr"] - before["psnr"] <= -1.0 or
                                       after["ssim"] - before["ssim"] <= -0.01)}
        rows.append(row)

    worst = min(rows, key=lambda row: row["delta_psnr"])
    panel(pilot_dir, worst["sample"], pilot_dir / "worst_psnr_case.png")
    lowest = min(rows, key=lambda row: row["dctta_psnr"])
    panel(pilot_dir, lowest["sample"], pilot_dir / "lowest_psnr_case.png")
    report = {"status": "PASS", "dataset": "Rain100L", "protocol": "seen operator; training-manifest Support Pool",
              "query_count": len(rows), "image_sizes_wh": sorted({(row["width"], row["height"]) for row in rows}),
              "mean_baseline_psnr": baseline["mean_psnr"], "mean_dctta_psnr": adapted_summary["mean_psnr"],
              "mean_delta_psnr": statistics.fmean(row["delta_psnr"] for row in rows),
              "mean_baseline_ssim": baseline["mean_ssim"], "mean_dctta_ssim": adapted_summary["mean_ssim"],
              "mean_delta_ssim": statistics.fmean(row["delta_ssim"] for row in rows),
              "psnr_improved_count": sum(row["delta_psnr"] > 0 for row in rows),
              "ssim_improved_count": sum(row["delta_ssim"] > 0 for row in rows),
              "baseline_peak_cuda_memory_bytes": baseline["peak_cuda_memory_bytes"],
              "dctta_peak_cuda_memory_bytes": adapted_summary["peak_cuda_memory_bytes"],
              "baseline_total_wall_seconds": baseline["total_wall_seconds"],
              "dctta_total_wall_seconds": adapted_summary["total_wall_seconds"],
              "baseline_total_evaluation_seconds": baseline["total_evaluation_seconds"],
              "dctta_fisher_seconds": adapted_summary["fisher_seconds"],
              "dctta_total_adaptation_seconds": adapted_summary["total_adaptation_seconds"],
              "dctta_total_evaluation_seconds": adapted_summary["total_evaluation_seconds"],
              "worst_psnr_case": worst["sample"],
              "lowest_psnr_case": lowest["sample"],
              "material_regression_cases": [row["sample"] for row in rows if row["material_regression"]],
              "rows": rows,
              "interpretation_limit": "Eight native-size test images and one fixed training Support pair; exploratory Pilot, not a formal benchmark or evidence of unseen-operator generalization."}
    (pilot_dir / "pilot_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    with open(pilot_dir / "pilot_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Rain100L 8-image Pilot", "",
             "Protocol: AWRaCLe training-manifest low-rain Support; eight Rain100L test Queries at native 320×480 or 480×320 dimensions. GT was used after restoration only.", "",
             f"Mean PSNR: {report['mean_baseline_psnr']:.4f} → {report['mean_dctta_psnr']:.4f} dB (Δ {report['mean_delta_psnr']:+.4f}); {report['psnr_improved_count']}/8 improved.",
             f"Mean SSIM: {report['mean_baseline_ssim']:.5f} → {report['mean_dctta_ssim']:.5f} (Δ {report['mean_delta_ssim']:+.5f}); {report['ssim_improved_count']}/8 improved.",
             f"Peak CUDA memory: baseline {report['baseline_peak_cuda_memory_bytes'] / 2**30:.2f} GiB; DCTTA {report['dctta_peak_cuda_memory_bytes'] / 2**30:.2f} GiB.",
             f"Wall time: baseline {report['baseline_total_wall_seconds']:.2f} s; DCTTA {report['dctta_total_wall_seconds']:.2f} s. DCTTA Fisher {report['dctta_fisher_seconds']:.2f} s, online adaptation {report['dctta_total_adaptation_seconds']:.2f} s, final evaluation {report['dctta_total_evaluation_seconds']:.2f} s.",
             "", "| Image | Baseline PSNR | DCTTA PSNR | Δ PSNR | Baseline SSIM | DCTTA SSIM | Δ SSIM | Adapt s |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['sample']} | {row['baseline_psnr']:.3f} | {row['dctta_psnr']:.3f} | {row['delta_psnr']:+.3f} | {row['baseline_ssim']:.4f} | {row['dctta_ssim']:.4f} | {row['delta_ssim']:+.4f} | {row['dctta_adaptation_seconds']:.2f} |")
    lines.extend(["", f"Worst PSNR change: {worst['sample']} ({worst['delta_psnr']:+.3f} dB). See `worst_psnr_case.png`.",
                  f"Lowest absolute DCTTA PSNR: {lowest['sample']} ({lowest['dctta_psnr']:.3f} dB). See `lowest_psnr_case.png`.",
                  f"Material regression heuristic (≤−1 dB or ≤−0.01 SSIM): {', '.join(report['material_regression_cases']) or 'none'}.",
                  "", report["interpretation_limit"]])
    (pilot_dir / "pilot_analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
