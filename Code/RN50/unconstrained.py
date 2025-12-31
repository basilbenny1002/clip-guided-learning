import torch
from torchvision import transforms
from torch.optim import Adam
from PIL import Image
import clip

# === Configuration ===
LEARNING_RATE = 0.05
STEPS = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on {DEVICE}...")

model, preprocess = clip.load("RN50", device=DEVICE)

normalize = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711)
)

def generate_unconstrained_image(prompt, model_name="RN50"):
    print(f"\n=== Generating (Unconstrained) for: '{prompt}' with model: {model_name} ===")
    
    text_token = clip.tokenize([prompt]).to(DEVICE)
    with torch.no_grad():
        text_emb = model.encode_text(text_token)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    image = torch.full((1, 3, 224, 224), 0.5, device=DEVICE)
    image = image + (torch.randn_like(image) * 0.1)
    image.requires_grad_(True)

    optimizer = Adam([image], lr=LEARNING_RATE)

    print(f"Optimizing noise for: '{prompt}' with {model_name}")

    for step in range(STEPS):
        optimizer.zero_grad()
        total_loss = 0
        for _ in range(4):
            aug_img = image
            image_norm = normalize(aug_img)
            img_emb = model.encode_image(image_norm)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            loss = (1 - torch.cosine_similarity(img_emb, text_emb).mean())
            total_loss += loss
        
        final_loss = total_loss / 4
        final_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            image.data.clamp_(0, 1)
            
        if step % 1000 == 0:
             with torch.no_grad():
                raw_emb = model.encode_image(normalize(image))
                raw_emb /= raw_emb.norm(dim=-1, keepdim=True)
                raw_sim = torch.cosine_similarity(raw_emb, text_emb).item()
             print(f"Step {step} | Loss: {final_loss.item():.4f} | Current Raw Score: {raw_sim:.4f}")

    filename = f"unconstrained_{model_name}_{prompt.replace(' ', '_')}.png"
    out_img = image.detach().cpu().squeeze()
    out_pil = transforms.ToPILImage()(out_img)
    out_pil.save(filename)
    print(f"Saved {filename}")
    
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
        generate_unconstrained_image(prompt)
