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

The comparison between these two approaches reveals a critical insight: **AI models don't rely primarily on actual visual parameters they exploit texture bias and statistical patterns**.

- The unconstrained version achieves high similarity scores with pure noise because it optimizes purely for feature alignment
- The constrained version, with realistic visual regularization, has lower similarity scores, suggesting texture patterns matter more to the model than semantic visual content

This experiment was conducted across multiple CLIP models (RN50, ViT-B, ViT-L 14) to observe the differences in how they perceive and optimize these patterns.

## Files Structure

The project is organized by format (Code or Notebooks) and then by Model Architecture:

```
Code/
    RN50/
    ViT-B/
    ViT-L 14/
Notebooks/
    RN50/
    ViT-B/
    ViT-L 14/
results/
```

- **`Code/`** - Python scripts for running locally
- **`Notebooks/`** - Jupyter notebooks for interactive running (e.g., VS Code, Colab)
- **`results/`** - Generated images organized by model name

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

You can choose to run the experiments using either the Python scripts or Jupyter Notebooks. Navigate to the specific model folder you wish to test.

### Running Python Scripts

1. Navigate to the desired model folder inside `Code/`.
2. Run the constrained or unconstrained script.

Example for RN50:
```bash
cd Code/RN50
python constrained.py
# or
python unconstrained.py
```

The scripts will generate images for a list of prompts and save them to the `results/` folder under the respective model name.

### Running Notebooks

1. Open the notebooks in `Notebooks/[Model_Name]/` using VS Code or Jupyter.
2. Run the cells sequentially.

## Customization

You can edit the `TEXT_PROMPTS` list in the scripts or notebooks to generate images for different text descriptions:

```python
TEXT_PROMPTS = [
    "A cute golden retriever",
    "A snow-covered mountain peak",
    "A bright red tomato"
]
```

Adjust `LEARNING_RATE`, `STEPS`, and `TV_WEIGHT` (constrained only) to fine-tune the generation process.

## Results

All generated outputs are saved in the `results/` folder, organized by the model architecture used.

## References

- CLIP: [Learning Transferable Models for Computational Linguistics](https://openai.com/research/clip)
- Read More: [Why AI Thinks This Static Is a Dog](https://medium.com/@basilbenny1002/why-ai-thinks-this-static-is-a-dog-exploring-the-semantic-gap-in-vision-models-fd75085de29e)
