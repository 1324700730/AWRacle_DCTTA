# Rain100L 8-image Pilot

Protocol: AWRaCLe training-manifest low-rain Support; eight Rain100L test Queries at native 320×480 or 480×320 dimensions. GT was used after restoration only.

Mean PSNR: 36.9775 → 37.1060 dB (Δ +0.1286); 7/8 improved.
Mean SSIM: 0.96118 → 0.96098 (Δ -0.00019); 2/8 improved.
Peak CUDA memory: baseline 2.02 GiB; DCTTA 31.26 GiB.
Wall time: baseline 10.76 s; DCTTA 44.53 s. DCTTA Fisher 11.12 s, online adaptation 23.47 s, final evaluation 1.75 s.

| Image | Baseline PSNR | DCTTA PSNR | Δ PSNR | Baseline SSIM | DCTTA SSIM | Δ SSIM | Adapt s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6.png | 34.053 | 34.149 | +0.096 | 0.9562 | 0.9559 | -0.0003 | 2.84 |
| 13.png | 38.631 | 38.775 | +0.144 | 0.9740 | 0.9734 | -0.0006 | 2.84 |
| 24.png | 23.620 | 23.836 | +0.216 | 0.9012 | 0.9012 | +0.0000 | 2.85 |
| 29.png | 39.082 | 39.283 | +0.201 | 0.9730 | 0.9737 | +0.0007 | 2.84 |
| 48.png | 37.493 | 37.589 | +0.096 | 0.9648 | 0.9643 | -0.0006 | 3.12 |
| 72.png | 43.930 | 43.911 | -0.019 | 0.9856 | 0.9855 | -0.0001 | 3.21 |
| 83.png | 37.020 | 37.230 | +0.211 | 0.9673 | 0.9668 | -0.0005 | 2.85 |
| 89.png | 41.991 | 42.075 | +0.084 | 0.9673 | 0.9671 | -0.0002 | 2.91 |

Worst PSNR change: 72.png (-0.019 dB). See `worst_psnr_case.png`.
Lowest absolute DCTTA PSNR: 24.png (23.836 dB). See `lowest_psnr_case.png`.
Material regression heuristic (≤−1 dB or ≤−0.01 SSIM): none.

Eight native-size test images and one fixed training Support pair; exploratory Pilot, not a formal benchmark or evidence of unseen-operator generalization.
