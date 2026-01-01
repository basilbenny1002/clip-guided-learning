import torch
from torchvision import transforms
from torch.optim import Adam
from PIL import Image
import clip
import numpy as np
import os

# === Configuration ===
LEARNING_RATE = 0.05
STEPS = 10000
TV_WEIGHT = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on {DEVICE}...")

# === 1. Load CLIP ===
model, preprocess = clip.load("RN50", device=DEVICE)

# === 2. Helper Functions ===

def get_tv_loss(img_tensor):
    """
    Calculates the difference between neighbor pixels.
    High value = messy noise. Low value = smooth blobs.
    """
    b, c, h, w = img_tensor.shape
    h_tv = torch.pow((img_tensor[:, :, 1:, :] - img_tensor[:, :, :h-1, :]), 2).sum()
    w_tv = torch.pow((img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :w-1]), 2).sum()
    return h_tv + w_tv

aug_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.3),
])

normalize = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711)
)

def generate_constrained_image(prompt, model_name="RN50"):
    print(f"\n=== Generating for: '{prompt}' with model: {model_name} ===")
    
    # Tokenize Text
    text_token = clip.tokenize([prompt]).to(DEVICE)
    with torch.no_grad():
        text_emb = model.encode_text(text_token)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # Initialize Image
    image = torch.full((1, 3, 224, 224), 0.5, device=DEVICE)
    image = image + (torch.randn_like(image) * 0.1)
    image.requires_grad_(True)

    optimizer = Adam([image], lr=LEARNING_RATE)

    best_loss = float('inf')
    best_image = None

    print(f"Optimizing for: '{prompt}' with {model_name}")

    for step in range(STEPS):
        optimizer.zero_grad()
        
        # Augmentation
        augmented_img = aug_transform(image)
        
        # CLIP Loss
        image_norm = normalize(augmented_img)
        img_emb = model.encode_image(image_norm)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        
        sim_loss = 1 - torch.cosine_similarity(img_emb, text_emb).mean()
        
        # TV Loss
        tv_loss = get_tv_loss(image) * TV_WEIGHT
        
        total_loss = sim_loss + tv_loss
        
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            image.data.clamp_(0, 1)
            
        current_sim_loss = sim_loss.item()
        if current_sim_loss < best_loss:
            best_loss = current_sim_loss
            best_image = image.detach().cpu().clone()
            
        if step % 1000 == 0:
            print(f"Step {step} | Sim Loss: {current_sim_loss:.4f} | TV Loss: {tv_loss.item():.4f}")
            
    if best_image is not None:
        filename = os.path.join("results", "RN50", f"constrained_{model_name}_{prompt.replace(' ', '_')}.png")
        out_img = best_image.squeeze()
        out_pil = transforms.ToPILImage()(out_img)
        out_pil.save(filename)
        print(f"Saved {filename}")
        
        # Verify
        verify_img = preprocess(Image.open(filename)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            v_img_feat = model.encode_image(verify_img)
            v_img_feat /= v_img_feat.norm(dim=-1, keepdim=True)
            final_sim = torch.cosine_similarity(v_img_feat, text_emb).item()
        print(f"Final Similarity: {final_sim:.4f}")

TEXT_PROMPTS = [
    "A cute golden retriever",
    "A snow-covered mountain peak",
    "A bright red tomato"
]

if __name__ == "__main__":
    for prompt in TEXT_PROMPTS:
        generate_constrained_image(prompt)
