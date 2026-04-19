# GANs on CIFAR-10

[![Python](https://img.shields.io/badge/Python-3.6%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarvesh1karandikar/GANs-on-CIFAR-10-Dataset/blob/master/GAN%20on%20CIFAR-10.ipynb)

A Deep Convolutional GAN (DCGAN) trained from scratch on the CIFAR-10 dataset, with activation maximization and image reconstruction capabilities built in.

---

## What This Does

This project trains a DCGAN to generate realistic 32x32 colour images that resemble the 10 object classes in CIFAR-10 (planes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks). Beyond basic generation, it also implements two analysis techniques:

- **Image reconstruction** — given a real image, find the latent code that best reproduces it through the generator
- **Activation maximization** — optimise a latent code so the discriminator rates the generated output as maximally real, surfacing what the network "thinks" a real image looks like

---

## Architecture

The model is a standard DCGAN implemented in TensorFlow 1.x, with the following design:

### Generator
Takes a 64-dimensional noise vector and upsamples it to a 32x32x3 image through four stages:

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| FC1 | Fully connected + reshape | 4 x 4 x 128 |
| Conv2 | Transposed conv (4x4, stride 2) | 8 x 8 x 64 |
| Conv3 | Transposed conv (4x4, stride 2) | 16 x 16 x 32 |
| Conv4 | Transposed conv (4x4, stride 2) | 32 x 32 x 3 |

Batch normalisation and Leaky ReLU (alpha=0.2) are applied after each layer except the last, which uses sigmoid activation.

### Discriminator
Takes a 32x32x3 image and outputs a single real/fake logit through three strided convolutions:

| Layer | Operation | Filters |
|-------|-----------|---------|
| Conv1 | Conv (4x4, stride 2) | 32 |
| Conv2 | Conv (4x4, stride 2) + BN | 64 |
| Conv3 | Conv (4x4, stride 2) + BN | 128 |
| FC4 | Fully connected | 1 |

### Training Details
- **Loss**: Binary cross-entropy (sigmoid) for both networks
- **Optimizer**: RMSProp (lr = 1e-4) for both generator and discriminator
- **Batch size**: 32
- **Epochs**: 20
- **Latent code size**: 64
- **Loss smoothing**: Exponential moving average (factor 0.95) for stable loss curves

### Reconstruction and Activation Maximization
Both techniques work by fixing the generator weights and back-propagating through the network into the latent code using Adam (lr = 1e-2) for 100 optimisation steps.

---

## Results

After 20 epochs of training on 50,000 CIFAR-10 images:

- **Generated samples**: 8x8 grids of 64 images sampled from random noise vectors, visualised at the end of each epoch. Early epochs produce blurry colour blobs; later epochs resolve into rough object shapes with recognisable colours and textures.
- **Discriminator loss**: Starts high, then decreases and stabilises, reflecting the discriminator learning to separate real from fake images.
- **Generator loss**: Initially low (the generator easily fools an untrained discriminator), then rises as the discriminator improves, before both losses settle into an adversarial equilibrium.
- **Reconstruction**: Given a test image, the model finds a latent code that re-generates it with average L2 reconstruction loss printed to console. Side-by-side grids show originals vs. reconstructions.
- **Activation maximization**: Starting from random codes, gradient ascent through the discriminator produces hallucinated images that score as maximally real — these often show prototypical textures and colour patches characteristic of common CIFAR-10 classes.

Pre-trained weights are saved in `model/dcgan.*` (TensorFlow checkpoint format).

---

## Tech Stack

| Component | Version |
|-----------|---------|
| Python | 3.6 |
| TensorFlow | 1.x (uses `tf.Session`, `tf.placeholder`) |
| NumPy | any compatible |
| Matplotlib | any compatible |
| Jupyter Notebook | any compatible |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sarvesh1karandikar/GANs-on-CIFAR-10-Dataset.git
cd GANs-on-CIFAR-10-Dataset
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: This project uses TensorFlow 1.x APIs. On modern machines, install `tensorflow==1.15` or run inside the provided Colab environment which handles this automatically.

### 3. Download the CIFAR-10 dataset

```bash
bash get_datasets.sh
```

This downloads the Python version of CIFAR-10 (~163 MB) and unpacks it to `data/cifar-10-batches-py/`.

### 4. Run in Jupyter

```bash
jupyter notebook "GAN on CIFAR-10.ipynb"
```

Execute cells top to bottom. Training (~20 epochs) takes roughly 30–60 minutes on CPU. Use a GPU or Google Colab for faster runs.

### 5. Load pre-trained weights (optional)

Skip training entirely by running Cell 5 or Cell 6, which load the saved checkpoint from the `model/` directory.

---

## Google Colab

Click the badge at the top of this README to open the notebook directly in Colab. Colab provides a free GPU and has TensorFlow 1.x available via:

```python
%tensorflow_version 1.x
```

---

## What a Live Demo Would Look Like

A portfolio-ready demo could be a small Gradio or Streamlit app that:

1. Exposes a slider for each of the 64 latent dimensions
2. Calls the trained generator in real time and displays the resulting 32x32 image
3. Includes a "Reconstruct" tab where users upload any small image and see what the GAN thinks it looks like in latent space
4. Shows a random-walk animation through latent space, smoothly morphing between generated images

This would run comfortably on a free Hugging Face Space (CPU tier) since inference through the generator is a handful of matrix multiplications on a tiny image.

---

## What I Learned / Key Insights

- **GAN training is a balancing act.** If the discriminator learns too fast, gradient signal to the generator collapses (vanishing gradients). RMSProp with a low learning rate (1e-4) keeps both networks moving at a comparable pace.
- **Batch normalisation is critical in the generator.** Removing it causes mode collapse almost immediately — the generator maps all noise vectors to the same output.
- **Leaky ReLU in the discriminator matters.** Standard ReLU creates sparse gradients in the discriminator that starve the generator of learning signal.
- **The latent space is smooth and meaningful.** The reconstruction and activation maximization experiments show that similar images cluster nearby in latent space, which is a sign the generator has learned a structured representation rather than memorising training examples.
- **TensorFlow 1.x `tf.Session` patterns.** This project is a good reference for understanding the explicit graph-build / session-run paradigm that underlies modern frameworks.
