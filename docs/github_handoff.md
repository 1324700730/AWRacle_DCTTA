# GitHub handoff

The GitHub repository contains the code, configuration, the small `wavelet.mat`, an eight-image Rain100L example, and historical Pilot evidence. **It does not contain the three large pretrained weights.** The project owner transfers those files directly to the student. No Git LFS installation is needed.

## Three files to transfer separately

Place each received file at the following path relative to the cloned project root. Keep the exact filename and case:

| Destination | Size | SHA-256 |
|---|---:|---|
| `assets/checkpoints/awracle_best.ckpt` | 1,028,021,119 bytes | `4a8e26e90ae00557e5614593920839dfe53eaac2bdfaebfd864dfbf562aa1aba` |
| `assets/clip/ViT-B-32.pt` | 353,976,522 bytes | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
| `models/dctta/assets/vgg19-dcbb9e9d.pth` | 574,673,361 bytes | `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44` |

The small `models/dctta/assets/wavelet.mat` and bundled example images are tracked in Git. The two `assets/` subdirectories contain `.gitkeep`, so the destinations exist immediately after cloning. From the project root, verify all bundled and transferred resources before running:

```bash
sha256sum -c assets/MANIFEST.sha256
```

If any line fails, check the destination and retransfer that file. The code uses the local CLIP path in `models/awracle/model.py`; the baseline/DCTTA YAML files point to the local AWRaCLe checkpoint, and DCTTA points to the local VGG19 weights.

For the Python environment, use the pinned project [`requirement.txt`](../requirement.txt) and [environment setup](environment_setup.md). The full server `pip freeze` is saved separately for provenance; it is not the recommended install file.

## Publishing the code repository

The exact three weight paths are in `.gitignore`. Before committing, inspect the staged file list; none of the three names above should appear:

```bash
git init
git add .
git diff --cached --name-only
git status --short
```

Do not force-add the weights. Full datasets, server scratch outputs and generated result folders are also excluded. The original AWRaCLe/DCTTA source checkouts and the server's CLIP cache are not needed for the bundled example.

## Experiment and provenance notes

The local example is a seen deraining Pilot, not a formal Seen/Unseen split. Its Support comes from AWRaCLe's training Derain manifest entry 0 and is separate from the Rain100L test Queries. GT is used for post-adaptation metrics only. Exact source selection and hashes are retained in `docs/pilot/selection_audit.json`; the server paths there are historical metadata.

The three weights will be shared privately. Before making the **code and example data** public, the project owner should confirm redistribution rights for the Rain100L sample and copied DCTTA/RDDM code. AWRaCLe's MIT license is included at `models/awracle/LICENSE`.

## Local handoff verification

With the three files placed at the exact paths above, the `awracle_dl` CUDA environment completed both README eight-image commands on 2026-10-13. Baseline generated eight PNGs with mean PSNR/SSIM **36.9774577135 / 0.9611751884**; DCTTA generated eight PNGs with **37.1060449255 / 0.9609807581**. Both pairs of averages match the historical Pilot summaries exactly. The DCTTA run recorded `gt_used_for_adaptation=false`. This confirms the current directory layout and example commands on this server; the student's own CUDA/PyTorch installation still needs to be compatible with their GPU.
