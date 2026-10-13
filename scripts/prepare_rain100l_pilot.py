"""Create a portable Rain100L pilot with training-only Support and copied images."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_size(path):
    with Image.open(path) as image:
        return list(image.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 16:
        parser.error("Pilot count must be in [1, 16]")

    train_root = args.train_root.resolve()
    test_root = args.test_root.resolve()
    train_manifest = train_root / "derain_train_low.json"
    test_manifest = test_root / "derain_test_rain100l.json"
    train_entries = json.loads(train_manifest.read_text())
    test_entries = json.loads(test_manifest.read_text())
    if len(test_entries) < args.count or not train_entries:
        raise ValueError("Pilot manifests have too few entries")

    # This pair is explicitly listed by AWRaCLe's low-rain training manifest.
    support_entry = train_entries[0]
    support_degraded = (train_root / support_entry["image_path"]).resolve(strict=True)
    support_clean = (train_root / support_entry["target_path"]).resolve(strict=True)
    support_paths = {support_degraded, support_clean}
    if any(train_root not in p.parents for p in support_paths):
        raise ValueError("Support escaped the training directory")
    support_hashes = {sha256(p) for p in support_paths}

    candidates = []
    test_hashes = set()
    for item in test_entries:
        query = (test_root / item["image_path"]).resolve(strict=True)
        ground_truth = (test_root / item["target_path"]).resolve(strict=True)
        if query.stem != ground_truth.stem or query.suffix.lower() != ".png":
            raise ValueError(f"Unpaired Rain100L entry: {item}")
        if query in support_paths or ground_truth in support_paths:
            raise ValueError("Training Support path overlaps Rain100L test set")
        test_hashes.update((sha256(query), sha256(ground_truth)))
        candidates.append((query, ground_truth))
    if support_hashes & test_hashes:
        raise ValueError("Training Support bytes match a Rain100L test image")

    selected = random.Random(args.seed).sample(candidates, args.count)
    selected.sort(key=lambda pair: int(pair[0].stem))
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new Pilot directory: {output}")
    (output / "query").mkdir(parents=True)
    (output / "gt").mkdir()
    (output / "support").mkdir()
    shutil.copy2(support_degraded, output / "support" / "degraded.jpg")
    shutil.copy2(support_clean, output / "support" / "clean.jpg")
    records = []
    for query, ground_truth in selected:
        if image_size(query) != image_size(ground_truth):
            raise ValueError(f"Mismatched Query/GT dimensions: {query}")
        shutil.copy2(query, output / "query" / query.name)
        shutil.copy2(ground_truth, output / "gt" / ground_truth.name)
        records.append({"id": query.stem, "query": str(query), "gt": str(ground_truth),
                        "size_wh": image_size(query), "query_sha256": sha256(query),
                        "gt_sha256": sha256(ground_truth)})
    manifest = {"protocol": "seen_support_pool", "split": "train",
                "source_manifest": str(train_manifest), "source_manifest_sha256": sha256(train_manifest),
                "pairs": [{"degraded": "support/degraded.jpg", "clean": "support/clean.jpg",
                           "task": "derain", "source_dataset": "AWRaCLe Train/Derain low",
                           "train_manifest_index": 0}]}
    (output / "support_pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    audit = {"dataset": "Rain100L", "pilot_count": args.count, "seed": args.seed,
             "test_manifest": str(test_manifest), "test_manifest_sha256": sha256(test_manifest),
             "train_manifest": str(train_manifest), "train_manifest_sha256": sha256(train_manifest),
             "support_degraded": str(support_degraded), "support_clean": str(support_clean),
             "support_degraded_sha256": sha256(support_degraded),
             "support_clean_sha256": sha256(support_clean),
             "support_source_is_training_manifest": True,
             "support_path_and_sha256_disjoint_from_all_test_images": True,
             "query_gt_pairs": records,
             "selection_method": "Python random.Random(seed).sample over the 100-entry source test manifest, then numeric filename sort",
             "image_handling": "Copied local files; native dimensions preserved; runners crop at most 15 right/bottom pixels to a multiple of 16"}
    (output / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "ids": [r["id"] for r in records],
                      "sizes_wh": [r["size_wh"] for r in records],
                      "support": [str(support_degraded), str(support_clean)]}, indent=2))


if __name__ == "__main__":
    main()
