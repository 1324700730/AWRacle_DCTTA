# Integration report

This is a historical server-side validation record. The example images are under `data/demo/`; the three large model weights must be transferred separately into `assets/` and `models/dctta/assets/`. Server paths and `experiments/` references below describe the original run and are not required for the bundled example. See [GitHub handoff](github_handoff.md).

## Outcome

This delivery establishes an isolated project, preserves the original AWRaCLe network source, adds a parity-checked baseline, and ports DCTTA RDDM/TRG, DIR, and AWRaCLe-specific Fisher-TIPS. The bounded one-Query DCTTA end-to-end smoke now passes with the original ImageNet VGG19 weights. It verifies integration mechanics only; it is not a restoration-quality benchmark.

| Requirement | Status | Evidence |
|---|---|---|
| AWRaCLe baseline implementation | PASS (bounded smoke) | `models/awracle/model.py`, `scripts/infer.py`; one real query restored and saved |
| Original checkpoint available | PASS (file presence/metadata) | `/home/16t/hjw_data/Awracle/checkpoints/best.ckpt`; epoch 99 / step 419200 |
| TRG | PASS (component smoke) | official RDDM 5-update online training and generated input tested; `experiments/dctta_component_smoke/trg.json` |
| DIR | PASS (bounded end-to-end smoke) | Original VGG19 loss, 8-view teacher, EMA update, and student update completed for one 64×64 Query |
| TIPS | PASS (component smoke) | 359 tensors / 10,780,904 optimized values; 6,468,420 selected (59.9989%); selected values restored exactly, frozen parameter versions unchanged, and 4,034,069 unselected values changed; `experiments/dctta_component_smoke/tips.json` |
| Teacher–student adaptation | PASS (bounded end-to-end smoke) | 8-view context-aware pseudo-label and EMA/DIR student update completed; `experiments/dctta_e2e_smoke_20261013/metrics.json` |
| DIR wavelet transform | PASS (component smoke) | source `rec2` transform yields finite `[1,4,32,32,3]`; `experiments/dctta_component_smoke/wavelet.json` |
| Baseline parity | PASS (bounded CPU/GPU and one native-resolution Query) | 64×64 CPU/GPU parity and 320×480 GPU parity yielded exact output equality; `experiments/pilot_rain100l_8/full_resolution_parity.json` |
| DCTTA end-to-end smoke | PASS (bounded, one Query) | GPU0; 64×64 Query; one DIR update; five online RAdam TRG updates and five-step DDIM; 3.16 GiB peak CUDA allocation; output and complete metadata in `experiments/dctta_final_smoke_20261013/` |
| Fresh-run reset and fixed-seed reproducibility | PASS (two independent processes) | Final PNG SHA-256 and all recorded adaptation values match exactly; `experiments/dctta_e2e_repro_20261013/reproducibility.json` |
| Per-run/operator evaluation summary | PASS (bounded baseline and TTA) | Both modes save per-sample metrics and `summary.json`; GT is loaded only after TTA adaptation. Final smoke summaries are under `experiments/baseline_final_smoke/` and `experiments/dctta_final_smoke_20261013/` |
| GT leakage safeguards | PASS by API inspection; smoke protocol labeled few-shot | Run-time adaptation has no Query GT input; default Seen Support mode validates a train-split manifest. Component smokes explicitly used a Rain100H target-paired Support pair as mechanics-only few-shot fixtures, not unseen/generalization evidence. |
| Rain100L native-resolution Pilot | PASS (8-image exploratory run) | Training-manifest Support and eight 320×480/480×320 Rain100L Queries; Baseline 36.9775 dB / 0.96118, DCTTA 37.1060 dB / 0.96098; full report in `docs/rain100l_pilot_report.md` |

## Runtime environment and limitation

The CPU and GPU baseline smoke each used one 64×64 crop from Rain100H `norain-1.png`, explicit support pair Rain100H `norain-2.png`, and checkpoint epoch 99 / step 419200. Support preprocessing matches the source BICUBIC resize, center crop, ToTensor and CLIP normalization. CPU PSNR/SSIM were 17.3520290318 / 0.3915967047; GPU values were 17.3517652395 / 0.3916079700. Both saved 64×64 outputs. Paired forwards through the untouched source `AWRaCLe` class and copied class yielded maximum and mean absolute error 0 and identical PSNR/SSIM on both devices. The exact model source files have matching SHA-256. CPU evidence is under `experiments/baseline_cpu_smoke/`; GPU evidence is under `experiments/baseline_gpu_smoke/`. The GPU baseline run used idle RTX 4090 GPU0 and a 0.16 per-process allocator cap (~8 GiB); the device was idle again after completion.



The default Python lacks PyTorch. The `awracle_dl` env has Python 3.10.20, torch 2.13.0+cu130 and CLIP; CUDA is available. No package was installed or upgraded. Baseline CPU/GPU parity and TRG, TIPS, Teacher support-hook/pseudo-label, and wavelet component smokes passed. After the user supplied the original ImageNet VGG19 weights at `models/dctta/assets/vgg19-dcbb9e9d.pth`, a bounded end-to-end DCTTA smoke completed on GPU0. It used one 64×64 Query, few-shot component Support, one DIR update, five TRG updates, and no Query GT during adaptation; adaptation loss was finite (2.00885), 4,045,261 student parameter elements changed, and peak CUDA allocation was 3,162,785,792 bytes. The final evaluation-only run supplied the matching cropped GT after adaptation and recorded PSNR 17.34784 and SSIM 0.39251; this single fixture result is not evidence of a quality gain. Current-code evidence and output are under `experiments/dctta_final_smoke_20261013/`, with the paired baseline summary under `experiments/baseline_final_smoke/`. A second fresh process from the same checkpoint and seed produced a byte-identical output PNG and identical adaptation values. This confirms the bounded integration/reset path only, not full-resolution or dataset behavior.

## Known gaps before student experiments

1. Compare the original AWRaCLe evaluation entrypoint and the integrated full data pipeline; native-resolution class-level parity now passes on one 320×480 Query.
2. Repeat native-resolution TIPS/DIR streams across more target domains, Support pairs and seeds. The current 8-Query stream passed but is exploratory.
3. Independently audit the adapted Fisher tensor names/masks and compare the RDDM port against the source Trainer on a fixed seed/input.
4. Populate and independently audit the real Seen Support Pool manifest, define strict Seen/Unseen split manifests, and run full-dataset batch evaluation before formal experiments.

The TRG/TIPS/Teacher component smoke fixtures use a target-paired Rain100H Support pair and are explicitly labeled `few_shot_component_smoke`; they validate code mechanics only. The integrated runner defaults to `seen_pool` and rejects an unmanifested support pair. No formal training or full-dataset evaluation was run.

The subsequent Rain100L Pilot used the AWRaCLe training-manifest Support pair `4324.jpg`, checked path and SHA-256 disjointness against the complete Rain100L test manifest, and completed eight native-resolution Queries in one persistent DCTTA stream. It is a seen deraining operator Pilot; see `docs/rain100l_pilot_report.md` for per-image changes, time, memory, and the handoff decision.
