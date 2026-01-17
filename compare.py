import torch
from PIL import Image
import clip
import os

# === Configuration ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FOLDER = "results"
REAL_IMAGES_FOLDER = os.path.join(RESULTS_FOLDER, "Real images")

# Prompts and their corresponding Real Image filenames
PROMPTS_MAPPING = {
    "A cute golden retriever": "cute golden retriever.png",
    "A snow-covered mountain peak": "Snow covered mountain peak.png",
    "A bright red tomato": "bright red tomato.png"
}

# Models configurations
# format: (Directory Name, Model Identifier in filename, CLIP Model Name)
MODELS_CONFIG = [
    ("RN50", "RN50", "RN50"),
    ("ViT-B", "ViT-B", "ViT-B/32"),
    ("ViT-L 14", "ViT-L-14", "ViT-L/14")
]

def calculate_score(model, preprocess, image_path, text_prompt):
    """
    Calculate similarity using the provided loaded model
    """
    if not os.path.exists(image_path):
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(DEVICE)
        text_input = clip.tokenize([text_prompt]).to(DEVICE)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
        return similarity
    except Exception as e:
        # print(f"Error processing {image_path}: {e}")
        return None

def main():
    # Store results: results[prompt][model_dir] = {type: score}
    results = {prompt: {} for prompt in PROMPTS_MAPPING}
    
    print(f"Running on {DEVICE}...")

    # Iterate through models, load one by one to save memory
    for dir_name, model_id, clip_model_name in MODELS_CONFIG:
        print(f"Loading model: {clip_model_name}...")
        try:
            model, preprocess = clip.load(clip_model_name, device=DEVICE)
        except Exception as e:
            print(f"Failed to load model {clip_model_name}: {e}")
            continue

        for prompt, real_img_filename in PROMPTS_MAPPING.items():
            results[prompt][dir_name] = {}
            safe_prompt = prompt.replace(" ", "_")

            # 1. Real Image (Evaluated with current model)
            real_img_path = os.path.join(REAL_IMAGES_FOLDER, real_img_filename)
            score_real = calculate_score(model, preprocess, real_img_path, prompt)
            results[prompt][dir_name]['Real Image'] = score_real

            # 2. Constrained
            constrained_filename = f"constrained_{model_id}_{safe_prompt}.png"
            constrained_path = os.path.join(RESULTS_FOLDER, dir_name, constrained_filename)
            score_const = calculate_score(model, preprocess, constrained_path, prompt)
            results[prompt][dir_name]['Constrained'] = score_const

            # 3. Unconstrained
            unconstrained_filename = f"unconstrained_{model_id}_{safe_prompt}.png"
            unconstrained_path = os.path.join(RESULTS_FOLDER, dir_name, unconstrained_filename)
            score_unconst = calculate_score(model, preprocess, unconstrained_path, prompt)
            results[prompt][dir_name]['Unconstrained'] = score_unconst
            
        # Clear specific model from memory
        del model
        del preprocess
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print Tables
    
    for prompt in PROMPTS_MAPPING:
        print(f"\nComparing for: '{prompt}'")
        print(f"{'='*80}")
        print(f"{'Source Model':<20} | {'Constrained':<15} | {'Unconstrained':<15}")
        print(f"{'-'*80}")
        
        # Rows for each generative model
        for dir_name, _, _ in MODELS_CONFIG:
            scores = results[prompt].get(dir_name, {})
            
            const_score = scores.get('Constrained')
            const_str = f"{const_score:.4f}" if const_score is not None else "N/A"
            
            unconst_score = scores.get('Unconstrained')
            unconst_str = f"{unconst_score:.4f}" if unconst_score is not None else "N/A"
            
            print(f"{dir_name:<20} | {const_str:<15} | {unconst_str:<15}")

        print(f"{'-'*80}")

        # Row for Real Image (All metrics)
        real_scores_str_parts = []
        for dir_name, _, _ in MODELS_CONFIG:
            scores = results[prompt].get(dir_name, {})
            r_score = scores.get('Real Image')
            r_val = f"{r_score:.4f}" if r_score is not None else "N/A"
            real_scores_str_parts.append(f"{dir_name}: {r_val}")
        
        real_row_content = ", ".join(real_scores_str_parts)
        print(f"{'Real Image':<20} | {real_row_content}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
