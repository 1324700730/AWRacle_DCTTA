# AWRaCLe-CLIP + DCTTA

Independent image-restoration project integrating the AWRaCLe-CLIP backbone with DCTTA's TRG, DIR and Fisher-TIPS online adaptation. The original source projects are not needed to run it. The three large pretrained weight files are delivered separately; after placing them at the paths below, the eight-image Rain100L Pilot can run locally. It is an exploratory **seen deraining** example, not evidence of a general quality improvement.

## Repository contents

| Path | Purpose |
|---|---|
| `models/awracle/model.py` | AWRaCLe-CLIP restoration model |
| `models/dctta/`, `adaptation/` | RDDM/TRG, DIR, Fisher-TIPS and losses |
| `assets/checkpoints/awracle_best.ckpt` | Original AWRaCLe restoration checkpoint; **copy separately** |
| `assets/clip/ViT-B-32.pt` | CLIP ViT-B/32 used by AWRaCLe; **copy separately** |
| `models/dctta/assets/vgg19-dcbb9e9d.pth` | ImageNet VGG19 for the DIR perceptual loss; **copy separately** |
| `models/dctta/assets/wavelet.mat` | DCTTA wavelet transform coefficients |
| `data/demo/` | Copied eight-image Rain100L Query/GT sample and a separate training Support pair |
| `configs/` | Portable baseline and DCTTA defaults |
| `requirement.txt` | Exact project runtime packages from the working `awracle_dl` environment |
| `docs/pilot/` | Compact historical Pilot evidence |

The two DCTTA PromptIR checkpoints are deliberately excluded: they cannot be loaded by AWRaCLe. TRG trains its RDDM generator online and needs no pretrained RDDM checkpoint. Full training and benchmark datasets are not included; the bundled example is enough to run the same eight-image protocol after a clone.

## Clone and place the three weights

The three large weights are ignored by Git and are **not** in the GitHub clone. Obtain them from the project owner and place them with these exact names under the cloned repository root:

```bash
git clone <repository-url>
cd AWRaCLe_DCTTA
# Copy the three received files to:
# assets/checkpoints/awracle_best.ckpt
# assets/clip/ViT-B-32.pt
# models/dctta/assets/vgg19-dcbb9e9d.pth
sha256sum -c assets/MANIFEST.sha256
```

The two `assets/` directories and `models/dctta/assets/` are created by the clone because they contain tracked metadata or the small `wavelet.mat`. The checksum command detects missing, truncated or wrong weights before a run. All model code loads these files directly from the project; no cache or source-server path is required. See [GitHub handoff](docs/github_handoff.md) for exact transfer and publishing instructions.

For Linux with an NVIDIA driver compatible with this server's CUDA 13 wheel set, recreate the project environment with:

```bash
conda create -n awracle_dl python=3.10.20 -y
conda activate awracle_dl
python -m pip install -r requirement.txt
python -c "import torch, torchvision, clip; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

The server Pilot used Python 3.10.20, torch 2.13.0+cu130 and torchvision 0.28.0+cu130 on a 48 GiB GPU. For another GPU, CUDA build or operating system, follow [environment setup](docs/environment_setup.md) to install a compatible torch/torchvision pair and then `requirements-core.txt`. The eight-image DCTTA peak PyTorch allocation was 31.26 GiB. A smaller GPU may need a lower-resolution or shorter exploratory run; that changes the experimental setting.

## Run the bundled example

Run commands from the repository root. The same training Support pair is used for both paths. GT is optional and used only after inference for metrics.

```bash
python scripts/infer.py \
  --config configs/baseline.yaml \
  --input-dir data/demo/query \
  --support-degraded data/demo/support/degraded.jpg \
  --support-clean data/demo/support/clean.jpg \
  --gt-dir data/demo/gt \
  --output-dir results/baseline_demo \
  --operator rain100l_seen_demo --device cuda:0
```

```bash
python scripts/run_tta.py \
  --config configs/dctta.yaml \
  --input-dir data/demo/query \
  --support-degraded data/demo/support/degraded.jpg \
  --support-clean data/demo/support/clean.jpg \
  --support-manifest data/demo/support_pool_manifest.json \
  --gt-dir data/demo/gt \
  --output-dir results/dctta_demo \
  --operator rain100l_seen_demo --device cuda:0
```

Each DCTTA command uses one target-domain stream: Fisher estimation on the Query pool, sequential persistent Teacher/Student/TRG adaptation, then evaluation with the final Student. Start a fresh process and choose a new empty output directory for each independent target domain. `--max-samples 1` bounds an exploratory run; `--max-samples 0` uses every image. `--disable-trg` and `--disable-tips` provide the two component ablations. The DCTTA runner requires a Seen Support Pool manifest by default. A target-paired experiment must explicitly use `--support-protocol few_shot` and be reported separately.

If the student's GPU cannot hold the full-resolution eight-image DCTTA run, first check the installation with the bundled 64×64 Query. This verifies loading and one complete adaptation step; it is not a substitute for full-resolution metrics:

```bash
python scripts/run_tta.py \
  --config configs/dctta.yaml \
  --input-dir data/demo/smoke \
  --support-degraded data/demo/support/degraded.jpg \
  --support-clean data/demo/support/clean.jpg \
  --support-manifest data/demo/support_pool_manifest.json \
  --output-dir results/dctta_smoke \
  --operator handoff_smoke --device cuda:0 --max-samples 1
```

## New data and handoff

For a new Rain100L sample, provide the AWRaCLe training Derain directory and Rain13K directory to `scripts/prepare_rain100l_pilot.py`; it checks Support overlap and **copies** selected images plus the training Support into a new portable folder:

```bash
python scripts/prepare_rain100l_pilot.py \
  --train-root /path/to/data_awracle/Train/Derain \
  --test-root /path/to/data_awracle/Rain13K \
  --output-dir data/my_pilot --count 8 --seed 23
```

Use `data/my_pilot/query`, `gt`, `support/degraded.jpg`, `support/clean.jpg` and `support_pool_manifest.json` in the commands above. Do not mix target test GT into adaptation. See [the Pilot report](docs/rain100l_pilot_report.md), [integration audit](docs/integration_audit.md), and [GitHub handoff notes](docs/github_handoff.md). The source server's full scratch outputs are excluded from Git; the compact Pilot evidence is under `docs/pilot/`.
