# CLAUDE.md — GANs on CIFAR-10

## Project Summary

A Deep Convolutional GAN (DCGAN) built with TensorFlow 1.x that learns to generate 32x32 RGB images resembling CIFAR-10 classes. The project is implemented as a single Jupyter notebook (`GAN on CIFAR-10.ipynb`) with helper utilities in `lib/`. A pre-trained TensorFlow checkpoint is stored in `model/`. The project also implements latent-space analysis: image reconstruction (find the code that best recovers a given image) and activation maximization (optimise a code so the discriminator rates it as maximally real).

## How to Run

### Prerequisites

- Python 3.6
- TensorFlow 1.x (`pip install tensorflow==1.15`)
- NumPy, Matplotlib (`pip install -r requirements.txt`)

### Steps

1. Download data: `bash get_datasets.sh` — places CIFAR-10 in `data/cifar-10-batches-py/`
2. Open notebook: `jupyter notebook "GAN on CIFAR-10.ipynb"`
3. Run all cells top-to-bottom for full training + visualization
4. OR skip to Cell 5/6 to load the pre-trained checkpoint from `model/dcgan.*` and run reconstruction/actmax only

On Google Colab, prepend `%tensorflow_version 1.x` before the imports cell.

## Model Architecture

### Generator (noise → image)

```
Input: noise vector [batch, 64]
  → FC(4*4*128) → reshape [batch, 4, 4, 128] → BN → LeakyReLU(0.2)
  → ConvTranspose(4x4, stride=2, 64 filters)   → [batch, 8, 8, 64]   → BN → LeakyReLU
  → ConvTranspose(4x4, stride=2, 32 filters)   → [batch, 16, 16, 32] → BN → LeakyReLU
  → ConvTranspose(4x4, stride=2, 3 filters)    → [batch, 32, 32, 3]  → Sigmoid
Output: generated image [batch, 32, 32, 3]
```

### Discriminator (image → real/fake logit)

```
Input: image [batch, 32, 32, 3]
  → Conv(4x4, stride=2, 32 filters)  → [batch, 16, 16, 32] → LeakyReLU(0.2)
  → Conv(4x4, stride=2, 64 filters)  → [batch, 8, 8, 64]   → BN → LeakyReLU
  → Conv(4x4, stride=2, 128 filters) → [batch, 4, 4, 128]  → BN → LeakyReLU
  → reshape [batch, 2048] → FC(1)
Output: logit scalar [batch, 1]
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent code size | 64 |
| Batch size | 32 |
| Epochs | 20 |
| Discriminator/Generator LR | 1e-4 (RMSProp) |
| Reconstruction/Actmax LR | 1e-2 (Adam) |
| Reconstruction steps | 100 |
| Actmax steps | 100 |
| Loss function | Binary cross-entropy (sigmoid) |

### Extra capabilities

- **Reconstruction**: For a given test image, optimises `actmax_code` over 100 Adam steps to minimise L2 distance between generator output and target. Reports average reconstruction loss.
- **Activation maximization**: Optimises `actmax_code` to maximise discriminator output (i.e., "most real" image). Reveals what features the discriminator has learned.

## Current Limitations

1. **TensorFlow 1.x only.** Uses `tf.Session`, `tf.placeholder`, `tf.get_variable` — incompatible with TF2 without `compat.v1` shim. Colab requires `%tensorflow_version 1.x` magic.
2. **No class conditioning.** The generator has no label input, so it cannot be directed to generate a specific CIFAR-10 class.
3. **32x32 output only.** The architecture is fixed to CIFAR-10 image size; no progressive growing or super-resolution.
4. **CPU training is slow.** Training for 20 epochs on CPU takes 30–60 min. A GPU is strongly recommended.
5. **No FID or IS metrics.** Quality is evaluated only visually — no Fréchet Inception Distance or Inception Score computed.
6. **Python 2/3 compatibility shim in `lib/datasets.py`** uses `cPickle` (Python 2), which conflicts with the main notebook that uses `pickle` directly. The `lib/` utilities (train.py, datasets.py) appear to be from an earlier RNN assignment and are not actually used by the GAN notebook.
7. **No requirements file checked in.** Dependencies must be inferred from imports.
8. **Checkpoint portability.** The saved `model/dcgan.*` is a TF1 checkpoint that requires the exact same graph definition to restore.

## Enhancement TODO List

### Quick wins (< 1 day each)
- [ ] Add `requirements.txt` with pinned versions (tensorflow==1.15, numpy, matplotlib)
- [ ] Add a Colab-friendly first cell: `%tensorflow_version 1.x` + dataset download via `wget`
- [ ] Add markdown cells to the notebook explaining each section
- [ ] Visualize the latent space interpolation between two noise vectors as a GIF
- [ ] Save sample grids as PNG files after each epoch rather than only displaying inline

### Medium lift (1–3 days each)
- [ ] Port to TensorFlow 2.x / PyTorch using `tf.keras` or `torch.nn` modules
- [ ] Add FID score computation at end of training (use `torch-fidelity` or `tensorflow-gan`)
- [ ] Add class-conditional generation (cGAN) so a label can steer the generator
- [ ] Build a Gradio/Streamlit demo app that exposes latent-space sliders for real-time generation
- [ ] Log losses to TensorBoard for cleaner training visualization

### Big lift (1+ week each)
- [ ] Upgrade to Progressive GAN or StyleGAN2 for higher quality outputs
- [ ] Add a WGAN-GP variant and compare sample quality vs. standard DCGAN
- [ ] Train on a higher-resolution dataset (e.g., CelebA 64x64) to demonstrate scalability
- [ ] Deploy the inference demo to Hugging Face Spaces with model weights stored via Git LFS

## Recommended Demo Tier

**Recommended: Medium Lift** — Build a Gradio app with the pre-trained checkpoint.

**Justification**: The pre-trained weights are already in the repo (`model/dcgan.*`), so no re-training is required. A Gradio interface that loads the checkpoint, accepts a 64-dim noise vector (or latent sliders), and renders the generated 32x32 image can be built in a few hours and deployed for free on Hugging Face Spaces. This is meaningfully more impressive than a static notebook for a portfolio, while still being achievable in a weekend without GPU access.
