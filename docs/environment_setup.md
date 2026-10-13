# Recreate the working Python environment

The project's installable [`requirement.txt`](../requirement.txt) pins the **runtime packages actually used by this integration** to versions installed in the working `awracle_dl` environment. On the server that means Python **3.10.20**, PyTorch **2.13.0+cu130**, torchvision **0.28.0+cu130**, and CUDA runtime **13.0**. The OpenAI CLIP package is pinned to Git commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. Installation needs Internet access to the Python package indexes and GitHub, plus a compatible NVIDIA driver.

## Linux with a driver supporting this CUDA 13 wheel set

From the cloned project root:

```bash
conda create -n awracle_dl python=3.10.20 -y
conda activate awracle_dl
python -m pip install -r requirement.txt
python -c "import torch, torchvision, clip; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
sha256sum -c assets/MANIFEST.sha256
```

The version check must show `2.13.0+cu130`, `0.28.0+cu130`, and `True` for CUDA before running the DCTTA example. The three model files in the checksum list are delivered privately and must be copied to the paths in [GitHub handoff](github_handoff.md) first. The example commands are in the [README](../README.md).

## Different GPU, CUDA version, or operating system

The CUDA 13 pins may not have a wheel for the student's platform or driver. In that case, create a Python 3.10 environment, install a **mutually compatible** torch/torchvision pair using the command for that machine at [PyTorch's installer](https://pytorch.org/get-started/locally/), then install the other exact project dependencies:

```bash
conda create -n awracle_dl python=3.10 -y
conda activate awracle_dl
# Install the matching torch and torchvision pair first.
python -m pip install -r requirements-core.txt
python -c "import torch, torchvision, clip; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
sha256sum -c assets/MANIFEST.sha256
```

This alternative changes the PyTorch build, so reproduce benchmark numbers only after recording the student's exact torch/torchvision/CUDA versions. Full-resolution eight-image DCTTA allocated about **31.26 GiB** of PyTorch GPU memory on the server; the README also includes a 64×64 installation check for smaller GPUs.

## Why the complete `pip freeze` is separate

[`awracle_dl_pip_freeze.txt`](awracle_dl_pip_freeze.txt) records all 82 installed packages for provenance. It includes unrelated research packages. The existing server environment's `pip check` reports missing optional dependencies of `basicsr` (`addict`, `future`, `lmdb`, `opencv-python`, `tb-nightly`, `yapf`) even though this integrated project runs successfully. Installing that complete snapshot would copy those unrelated inconsistencies, so use `requirement.txt` for this project.
