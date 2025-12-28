import torch
from torchvision import transforms
from torch.optim import Adam
from PIL import Image
import clip
import numpy as np
import random
import torch.nn.functional as F

# === Configuration ===
TEXT_PROMPT = "A cute golden retriever"
LEARNING_RATE = 0.05
STEPS = 10000  # Same steps as constrained
TV_WEIGHT = 1e-4  # Controls how "smooth" the image is
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on {DEVICE}...")

# === 1. Load CLIP ===
# CHANGED: Switched from B/32 to L/14
model, preprocess = clip.load("ViT-L/14", device=DEVICE)

# === 2. Helper Functions ===

# Total Variation Loss (Force Smoothness)
def get_tv_loss(img_tensor):
    """
    Calculates the difference between neighbor pixels.
    High value = messy noise. Low value = smooth blobs.
    """
    b, c, h, w = img_tensor.shape
    h_tv = torch.pow((img_tensor[:, :, 1:, :] - img_tensor[:, :, :h-1, :]), 2).sum()
    w_tv = torch.pow((img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :w-1]), 2).sum()
    return h_tv + w_tv

# Augmentation Pipeline (Force Robustness)
aug_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.3),
])

# Normalization for CLIP
normalize = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711)
)



# Tokenize Text
text_token = clip.tokenize([TEXT_PROMPT]).to(DEVICE)
with torch.no_grad():
    text_emb = model.encode_text(text_token)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# Initialize Image (Start with Grey instead of pure noise for better convergence)
image = torch.full((1, 3, 224, 224), 0.5, device=DEVICE) # Grey background
image = image + (torch.randn_like(image) * 0.1) # Add slight noise
image.requires_grad_(True)

optimizer = Adam([image], lr=LEARNING_RATE)

# === 4. Optimization Loop ===
best_loss = float('inf')
best_image = None

print(f"Optimizing for: '{TEXT_PROMPT}'")

for step in range(STEPS):
    optimizer.zero_grad()

    # A. Apply Augmentations (Crucial for robustness)
    # We clone to avoid in-place errors during gradient calculation
    augmented_img = aug_transform(image) #Uncommented for constrained

    # B. CLIP Loss
    image_norm = normalize(augmented_img)
    img_emb = model.encode_image(image_norm)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    # We want similarity to be 1, so loss is 1 - similarity
    sim_loss = 1 - torch.cosine_similarity(img_emb, text_emb).mean()

    # C. TV Loss (Smoothness)
    tv_loss = get_tv_loss(image) * TV_WEIGHT

    # Total Loss
    total_loss = sim_loss + tv_loss

    total_loss.backward()
    optimizer.step()

    # Clamp image to keep colors valid (0 to 1)
    with torch.no_grad():
        image.data.clamp_(0, 1)

    # Track Best (Using pure similarity, ignoring TV for the "best" metric)
    current_sim_loss = sim_loss.item()
    if current_sim_loss < best_loss:
        best_loss = current_sim_loss
        best_image = image.detach().cpu().clone()

    if step % 100 == 0:
        print(f"Step {step} | Sim Loss: {current_sim_loss:.4f} | TV Loss: {tv_loss.item():.4f}")

# === 5. Save Output === Here we save the image with the lowest loss
print(f"Optimization finished. Best Similarity Loss: {best_loss:.4f}")

if best_image is not None:
    out_img = best_image.squeeze()
    out_pil = transforms.ToPILImage()(out_img)
    # CHANGED: Updated filename to include model name (replaced slash with dash)
    out_pil.save("results/constrained result ViT-L-14.png")

    print("\nComparing the similarity")

    # CHANGED: Updated filename here as well to match
    verify_img = preprocess(Image.open("results/constrained result ViT-L-14.png")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        v_img_feat = model.encode_image(verify_img)
        v_img_feat /= v_img_feat.norm(dim=-1, keepdim=True)

        # Recalculate text emb just in case
        v_text_feat = model.encode_text(text_token)
        v_text_feat /= v_text_feat.norm(dim=-1, keepdim=True)

        final_sim = torch.cosine_similarity(v_img_feat, v_text_feat).item()

    print(f"Final Similarity Score for constrained : {final_sim:.4f}")