import torch
from torchvision import transforms
from torch.optim import Adam
from PIL import Image
import clip
import copy
# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text = "A beautiful potato"
text_token = clip.tokenize([text]).to(device)


def compare():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 2. Load the generated image and the original text
    # Ensure the image filename matches what you saved earlier
    image_path = "best_generated_image.png"
    

    try:
        # Preprocess image (resize, normalize, convert to tensor)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        # Tokenize text
        text = clip.tokenize([text]).to(device)

        # 3. Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_token)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = torch.cosine_similarity(image_features, text_features).item()

        print(f"Text: '{text}'")
        print(f"CLIP Confidence/Similarity Score: {similarity:.4f}")


    except FileNotFoundError:
        print(f"Error: Could not find '{image_path}'. Make sure the file exists.")





with torch.no_grad():
    text_emb = model.encode_text(text_token)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

# === Step 2: Initialize a random image tensor ===
image = torch.randn((1, 3, 224, 224), device=device, requires_grad=True)
optimizer = Adam([image], lr=0.05)

# Normalization for CLIP
normalize = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711)
)


# Initialize tracking variables
best_loss = float('inf')
best_image = None

# === Step 3: Optimize image ===
print("Starting optimization...")
for step in range(10000):
    optimizer.zero_grad()
    image_norm = normalize(image)
    img_emb = model.encode_image(image_norm)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    # Loss: maximize cosine similarity
    loss = 1 - torch.cosine_similarity(img_emb, text_emb).mean()
    loss.backward()
    optimizer.step()

    # --- NEW: Track and save the best image ---
    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss
        # We must clone/detach to save the actual pixel values at this moment
        best_image = image.detach().cpu().clone() 
        
        # Optional: Print when a new record is set
        # print(f"New best found at step {step}: {best_loss:.5f}")

    if step % 50 == 0:
        print(f"Step {step} | Loss: {current_loss:.4f} | Best So Far: {best_loss:.4f}")

# === Step 4: Save the BEST output ===
if best_image is not None:
    print(f"Saving image with lowest loss: {best_loss:.5f}")
    # Process the 'best_image' we saved during the loop
    out_img = best_image.squeeze().clamp(0,1)
    out_pil = transforms.ToPILImage()(out_img)
    out_pil.save("best_generated_image.png")
else:
    print("Something went wrong, no best image found.")