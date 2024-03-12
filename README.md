# GAN for MNIST Image Generation

## Overview
This project utilizes a Generative Adversarial Network (GAN) to generate digit images similar to those found in the MNIST dataset. It consists of two main components: a Generator and a Discriminator. The Generator learns to generate new images that resemble the handwritten digits, while the Discriminator learns to differentiate between real images from the MNIST dataset and fake images produced by the Generator.

## Installation

### Prerequisites
- Python 3.6 or higher
- PyTorch 1.7.1 or higher
- torchvision 0.8.2 or higher

## How to Run

To play the game, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/lileetung/mnist-GAN.git
```

2. Navigate to the cloned repository:

```bash
cd mnist-GAN
```

3. Run the game:

```bash
python gan.py
```