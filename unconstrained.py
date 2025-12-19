import torch
from torchvision import transforms
from torch.optim import Adam
from PIL import Image
import clip
import torch.nn.functional as F

# === Configuration ===
TEXT_PROMPT = "A cute golden retriever"
LEARNING_RATE = 0.05
STEPS = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on {DEVICE}...")

#Load teh clip model
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# # === 2. Define Augmentations ) === ## NOT BEIGN USED FOR UNCONSTRAINED CASE
# aug_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
#     transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
# ])

# Normalization for CLIP
normalize = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711)
)

# tokenizing the text 
text_token = clip.tokenize([TEXT_PROMPT]).to(DEVICE)
with torch.no_grad():
    text_emb = model.encode_text(text_token)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# Initialize Image: Start with Grey instead of pure noise 
image = torch.full((1, 3, 224, 224), 0.5, device=DEVICE)
image = image + (torch.randn_like(image) * 0.1)
image.requires_grad_(True)


optimizer = Adam([image], lr=LEARNING_RATE)

# === 4. Optimization Loop ===
print(f"Optimizing noise for: '{TEXT_PROMPT}'")

for step in range(STEPS):
    optimizer.zero_grad()

    total_loss = 0
    for _ in range(4):
        # aug_img = aug_transform(image) #Commented out to give rise to random noises
        aug_img = image
        # Normalize and Encode
        image_norm = normalize(aug_img)
        img_emb = model.encode_image(image_norm)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # Loss: 1 - Similarity
        loss = (1 - torch.cosine_similarity(img_emb, text_emb).mean())
        total_loss += loss

    # Average the loss over the augmentations
    final_loss = total_loss / 4
    final_loss.backward()

    optimizer.step()

    # Clamp to valid image range (0-1)
    with torch.no_grad():
        image.data.clamp_(0, 1)

    if step % 100 == 0:
        # Check the RAW similarity (without augmentation) to see true progress
        with torch.no_grad():
            raw_emb = model.encode_image(normalize(image))
            raw_emb /= raw_emb.norm(dim=-1, keepdim=True)
            raw_sim = torch.cosine_similarity(raw_emb, text_emb).item()

        print(f"Step {step} | Loss: {final_loss.item():.4f} | Current Raw Score: {raw_sim:.4f}")

# Saving teh image and checking similarity
print("\n--- Saving ---")
out_img = image.detach().cpu().squeeze()
out_pil = transforms.ToPILImage()(out_img)
out_pil.save("results/unconstrained result.png")

print("Calculating similarity of the unconstrained image")
verify_img = preprocess(Image.open("unconstrained result.png")).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    v_img_feat = model.encode_image(verify_img)
    v_img_feat /= v_img_feat.norm(dim=-1, keepdim=True)

    final_sim = torch.cosine_similarity(v_img_feat, text_emb).item()

print(f"Final Similarity Score of the unconstrained image: {final_sim:.4f}")