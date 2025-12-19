# CLIP Guided Learning

A research exploration into how AI models understand and perceive images using CLIP (Contrastive Language-Image Pre-training).

## Overview

This repository contains two approaches to image generation guided by text descriptions:

### **Unconstrained Version**
- Starts with a grey image with slight noise
- Uses gradient descent to optimize the image to match the text description
- **Result**: Static noise pattern with high similarity score to the text prompt

### **Constrained Version**
- Starts with a grey image with slight noise
- Uses gradient descent with augmentations and Total Variation (TV) loss
- TV loss encourages smoother, more visually coherent images
- Augmentations force robustness across transformations
- **Result**: Visual patterns with lower (but more realistic) confidence scores

## Key Findings

The comparison between these two approaches reveals a critical insight: **AI models don't rely primarily on actual visual parameters—they exploit texture bias and statistical patterns**. 

- The unconstrained version achieves high similarity scores with pure noise because it optimizes purely for feature alignment
- The constrained version, with realistic visual regularization, has lower similarity scores, suggesting texture patterns matter more to the model than semantic visual content

## Files

- **`unconstrained.py`** - Unconstrained image generation script
- **`constrained.py`** - Constrained image generation with TV loss and augmentations
- **`unconstrained.ipynb`** - Jupyter notebook version for Google Colab
- **`constrained.ipynb`** - Jupyter notebook version for Google Colab
- **`results/`** - Generated images (example results from "A cute golden retriever" prompt)

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow (PIL)
- openai-clip

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

**Unconstrained version:**
```bash
python unconstrained.py
```

**Constrained version:**
```bash
python constrained.py
```

Both scripts will generate an image and save it to the `results/` folder with a similarity score.

### Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Click **File → Open notebook**
3. Go to the **GitHub** tab
4. Enter: `basilbenny1002/clip_guided_learning`
5. Open either `unconstrained.ipynb` or `constrained.ipynb`
6. Run the cells sequentially (the first cell installs required packages)
7. Check the results folder for generated images

## Customization

Edit the `TEXT_PROMPT` variable in either script or notebook to generate images for different text descriptions:

```python
TEXT_PROMPT = "Your description here"
```

Adjust `LEARNING_RATE`, `STEPS`, and `TV_WEIGHT` (constrained only) to fine-tune the generation process.

## Results

Example results are provided in the `results/` folder, generated from the prompt: **"A cute golden retriever"**

- **unconstrained result.png** - High similarity but visually random noise
- **constrained result.png** - Lower similarity but with visible visual patterns

## References

- CLIP: [Learning Transferable Models for Computational Linguistics](https://openai.com/research/clip)
