# Rain100L native-resolution Pilot (8 Queries)

## Scope and data provenance

This is an exploratory **seen deraining operator** Pilot on the Rain100L benchmark. Rain100L uses synthetic rain on natural images; this run tests native image dimensions, not in-the-wild camera rain or an unseen operator. The eight Query IDs were sampled from the 100-entry AWRaCLe Rain100L test manifest with Python `random.Random(23).sample`: **6, 13, 24, 29, 48, 72, 83, 89**. Their dimensions are 480×320 or 320×480 pixels; both are multiples of 16, so the runners kept every pixel. The historical server run used symlinks; the handoff copy in `data/demo/` contains regular image files.

Both modes used the same AWRaCLe `best.ckpt` and the same degraded/clean Support pair, `Train/Derain/input/4324.jpg` and `Train/Derain/target/4324.jpg`. The pair is entry 0 of AWRaCLe's `derain_train_low.json`; the new project's Seen Support Pool manifest declares `split=train`. The Support paths and SHA-256 bytes were checked against **all 100** Rain100L test Query and GT images, with no exact overlap. The full selection, hashes, source manifest paths and selected image sizes are in [`selection_audit.json`](pilot/selection_audit.json). This check does not prove the absence of visually similar scenes or independently establish the checkpoint's exact training set.

For DCTTA, Fisher was estimated over all eight unlabeled Queries, one persistent student/teacher/TRG stream adapted sequentially over the seeded Query order, and the final adapted student evaluated all eight. GT was loaded for metrics after adaptation only. [`metrics.json`](pilot/metrics.json) records the Fisher/adaptation orders, parameter statistics, source commits, Support manifest hash, and `gt_used_for_adaptation=false`. A fresh process and the base checkpoint define an independent episode.

## Results

| Metric | AWRaCLe | AWRaCLe + DCTTA | Change |
|---|---:|---:|---:|
| Mean PSNR | 36.9775 dB | 37.1060 dB | +0.1286 dB |
| Mean SSIM | 0.96118 | 0.96098 | −0.00019 |
| End-to-end wall time, 8 images | 10.76 s | 44.53 s | 4.14× |
| Peak PyTorch CUDA allocated memory | 2.02 GiB | 31.26 GiB | 15.45× |

PSNR increased on 7/8 images. SSIM increased on 2/8. DCTTA spent 11.12 s estimating Fisher, 23.47 s in online adaptation, and 1.75 s in final evaluation. The baseline's eight evaluation passes took 2.96 s. End-to-end wall time includes startup and model loading; the measured times are descriptive for this server and run.

| Image | Baseline PSNR | DCTTA PSNR | Δ PSNR | Baseline SSIM | DCTTA SSIM | Δ SSIM |
|---|---:|---:|---:|---:|---:|---:|
| 6.png | 34.053 | 34.149 | +0.096 | 0.9562 | 0.9559 | −0.0003 |
| 13.png | 38.631 | 38.775 | +0.144 | 0.9740 | 0.9734 | −0.0006 |
| 24.png | 23.620 | 23.836 | +0.216 | 0.9012 | 0.9012 | +0.0000 |
| 29.png | 39.082 | 39.283 | +0.201 | 0.9730 | 0.9737 | +0.0007 |
| 48.png | 37.493 | 37.589 | +0.096 | 0.9648 | 0.9643 | −0.0006 |
| 72.png | 43.930 | 43.911 | −0.019 | 0.9856 | 0.9855 | −0.0001 |
| 83.png | 37.020 | 37.230 | +0.211 | 0.9673 | 0.9668 | −0.0005 |
| 89.png | 41.991 | 42.075 | +0.084 | 0.9673 | 0.9671 | −0.0002 |

All eight outputs were saved at native dimensions. The adaptation loss and gradients remained finite, all eight student updates changed parameters, and the process exited with code 0. The largest PSNR decrease was 72.png (−0.019 dB); the lowest absolute PSNR was 24.png (23.836 dB). Side-by-side panels are [`worst_psnr_case.png`](pilot/worst_psnr_case.png) and [`lowest_psnr_case.png`](pilot/lowest_psnr_case.png). No case met the exploratory regression flag of at least 1 dB PSNR or 0.01 SSIM loss. The restored images are visually close to baseline in these panels; this observation is limited to the displayed scale.

The original AWRaCLe class and integrated copy also gave **exactly identical** output on one 320×480 Pilot Query, with PSNR and SSIM differences of zero; see [`full_resolution_parity.json`](pilot/full_resolution_parity.json). Baseline and DCTTA summaries and per-image metrics are retained in `docs/pilot/`. Full scratch outputs, logs and process files remain only on the original server.

## Handoff decision

The integrated project is stable on this bounded native-resolution eight-image stream and can be handed to the student for **formal experiment design and execution**. This Pilot does not establish that DCTTA improves restoration quality. The mean PSNR gain is small, mean SSIM is slightly lower, and only one Support pair and one seed/order were used. The 31.26 GiB PyTorch peak allocation also makes GPU capacity a practical constraint; the run used a 48 GiB RTX 4090.

Before reporting scientific conclusions, independently verify the exact checkpoint's training data provenance, define strict Seen distribution shift versus unseen operator splits, repeat over more Queries, Support pairs, seeds and target domains, and run the planned ablations. Keep each independent target-domain stream in a fresh process and output directory. Do not use target test GT for adaptation or hyperparameter selection.
