# Multimodal UMAP

This repo originated as a challenge -- learn rich crossmodal representations of data by any means except neural networks. To do so, this project uses the UMAP dimensionality reduction algorithm to project data points into a shared latent space, in the hopes that the resulting representation is rich enough to use for cross-modal reconstruction. In effect, this project implements an autoencoder without the use of neural networks. Built primarily as a personal challenge, not polished enough for public use. For more information see the blog post.

## Installation and Usage

To get started, clone the repo and install the following dependencies:

```bash
pip install tqdm numpy torch torchvision transformers diffusers datasets matplotlib
```

Then run `main.py` to train the model on flickr30k and evaluate the the results on a few metrics. To view the adjustable training parameters, enter `python main.py --help` in the terminal and adjust accordingly, sensible defaults have been chosen. Note that the training process may take a while, especially on older CPUs!

## Files

- `impl/model.py` - Contains a basic UMAP implementation alongside a cross-modal extension
- `impl/dataset.py` - Loads in flickr30k and preprocesses captions using BERT and images using SD-VAE respectively
- `impl/util.py` - Contains wrapper functions for training and "inference"
- `impl/validation.py` - Contains implementations of kNN and cosine similarity metrics
- `impl/crossmodal.py` - Attempts to reconstruct images from captions using the cross-modal UMAP representation
