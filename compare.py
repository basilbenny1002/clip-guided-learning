import torch
from PIL import Image
import clip
import os

# === Configuration ===
TEXT_PROMPT = "A cute golden retriever"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FOLDER = "results"

# Image file paths
IMAGE = "image.png"
CONSTRAINED_IMAGE = os.path.join(RESULTS_FOLDER, "constrained result.png")
UNCONSTRAINED_IMAGE = os.path.join(RESULTS_FOLDER, "unconstrained result.png")

print(f"Running on {DEVICE}...")

# === Load CLIP ===
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# === Helper Functions ===

def get_image_text_similarity(image_path, text_prompt):
    """
    Calculate similarity between an image and a text prompt.
    Returns a similarity score between 0 and 1.
    """
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)
    
    text_input = clip.tokenize([text_prompt]).to(DEVICE)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity (cosine similarity)
        similarity = (image_features @ text_features.T).item()
    
    return similarity


# === Main Comparison ===

print(f"\nComparing images with text prompt: '{TEXT_PROMPT}'")
print(f"{'='*60}\n")

if os.path.exists(IMAGE):
    similarity = get_image_text_similarity(IMAGE, TEXT_PROMPT)
    print(f"Actual image similarity: {similarity:.4f}")
else:
    print(f"Actual image not found at: {IMAGE}")

if os.path.exists(CONSTRAINED_IMAGE):
    similarity = get_image_text_similarity(CONSTRAINED_IMAGE, TEXT_PROMPT)
    print(f"Constrained image similarity: {similarity:.4f}")
else:
    print(f"Constrained image not found at: {CONSTRAINED_IMAGE}")

if os.path.exists(UNCONSTRAINED_IMAGE):
    similarity = get_image_text_similarity(UNCONSTRAINED_IMAGE, TEXT_PROMPT)
    print(f"Unconstrained image similarity: {similarity:.4f}")
else:
    print(f"Unconstrained image not found at: {UNCONSTRAINED_IMAGE}")

print(f"{'='*60}")
