# # import torch
# # from torchvision import transforms
# # from torch.optim import Adam
# # from PIL import Image
# # import clip

# # # Load CLIP
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model, preprocess = clip.load("ViT-B/32", device=device)

# # # === Step 1: Get text embedding ===
# # text = "a cute golden retriever playing in a park"
# # text_token = clip.tokenize([text]).to(device)
# # with torch.no_grad():
# #     text_emb = model.encode_text(text_token)
# #     text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# # # === Step 2: Initialize a random image tensor ===
# # image = torch.randn((1, 3, 224, 224), device=device, requires_grad=True)
# # optimizer = Adam([image], lr=0.05)

# # # Normalization for CLIP
# # normalize = transforms.Normalize(
# #     mean=(0.48145466, 0.4578275, 0.40821073),
# #     std=(0.26862954, 0.26130258, 0.27577711)
# # )

# # # === Step 3: Optimize image so its CLIP embedding matches the text ===
# # for step in range(50000):
# #     optimizer.zero_grad()
# #     image_norm = normalize(image)
# #     img_emb = model.encode_image(image_norm)
# #     img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    
# #     # Loss: maximize cosine similarity
# #     loss = 1 - torch.cosine_similarity(img_emb, text_emb).mean()
# #     loss.backward()
# #     optimizer.step()
    
# #     if step % 50 == 0:
# #         print(f"Step {step} | Loss: {loss.item():.4f}")

# # # === Step 4: Save output ===
# # out_img = image.detach().cpu().squeeze().clamp(0,1)
# # out_pil = transforms.ToPILImage()(out_img)
# # out_pil.save("generated_from_text_embedding.png")


# import torch
# from torchvision import transforms
# from torch.optim import Adam
# from PIL import Image
# import clip

# # Load CLIP
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# model.eval()  # Ensure eval mode for speed

# # === Step 1: Get text embedding ===
# text = "a cute golden retriever playing in a park"
# text_token = clip.tokenize([text]).to(device)
# with torch.no_grad():
#     text_emb = model.encode_text(text_token)
#     text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# # === Step 2: Initialize a random image tensor ===
# image = torch.randn((1, 3, 224, 224), device=device, requires_grad=True)
# optimizer = Adam([image], lr=0.1)  # Higher LR
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)  # Decay LR

# # Normalization for CLIP
# normalize = transforms.Normalize(
#     mean=(0.48145466, 0.4578275, 0.40821073),
#     std=(0.26862954, 0.26130258, 0.27577711)
# )

# # === Step 3: Optimize image so its CLIP embedding matches the text ===
# scaler = torch.cuda.amp.GradScaler()  # For mixed precision
# for step in range(2000):  # Reduced steps
#     optimizer.zero_grad()
#     with torch.cuda.amp.autocast():  # Mixed precision
#         image_norm = normalize(image)
#         img_emb = model.encode_image(image_norm)
#         img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        
#         # Loss: maximize cosine similarity
#         loss = 1 - torch.cosine_similarity(img_emb, text_emb).mean()
#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()
#     scheduler.step()
    
#     if step % 50 == 0:
#         print(f"Step {step} | Loss: {loss.item():.4f}")

# # === Step 4: Save output ===
# out_img = image.detach().cpu().squeeze().clamp(0,1)
# out_pil = transforms.ToPILImage()(out_img)
# out_pil.save("generated_from_text_embedding.png")

import torch
from torchvision import transforms
from torch.optim import Adam
from PIL import Image
import clip

# Load CLIP (larger model for better quality)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)  # Changed to larger model
model.eval()

# === Step 1: Get text embedding ===
text = "a cute golden retriever playing in a park"
text_token = clip.tokenize([text]).to(device)
with torch.no_grad():
    text_emb = model.encode_text(text_token)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# === Step 2: Initialize a random image tensor (with blur for better start) ===
image = torch.randn((1, 3, 224, 224), device=device)
# Add slight blur to reduce noise
blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
image = blur(image)
image.requires_grad_(True)

# Save initial noise image
initial_img = image.detach().cpu().squeeze().clamp(0,1)
initial_pil = transforms.ToPILImage()(transforms.Resize((512, 512))(initial_img))
initial_pil.save("initial_noise.png")
optimizer = Adam([image], lr=0.05)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

# Normalization for CLIP
normalize = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711)
)

# === Step 3: Optimize image ===
scaler = torch.cuda.amp.GradScaler()
for step in range(100):  # Increased steps
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        image_norm = normalize(image)
        img_emb = model.encode_image(image_norm)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        
        # Loss: cosine similarity + regularization (reduce noise)
        cos_loss = 1 - torch.cosine_similarity(img_emb, text_emb).mean()
        reg_loss = 0.01 * torch.mean(image**2)  # Penalize large values
        loss = cos_loss + reg_loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    
    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

# === Step 4: Save output (upsample for better quality) ===
out_img = image.detach().cpu().squeeze().clamp(0,1)
out_pil = transforms.ToPILImage()(transforms.Resize((512, 512))(out_img))  # Upsample
out_pil.save("meow.png")