import json
import os
import random

import matplotlib.pyplot as plt
import torch
from PIL import Image

try:
    import open_clip
except ImportError as exc:
    raise SystemExit(
        "Missing open_clip. Install with: python -m pip install open_clip_torch"
    ) from exc


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, "tutorial_runs", "behavioral_cloning_data", "IMG")
    out_dir = os.path.join(base_dir, "tutorial_runs", "output")
    os.makedirs(out_dir, exist_ok=True)

    # Sample a small set of images for quick scoring
    all_imgs = [p for p in os.listdir(img_dir) if p.lower().endswith(".jpg")]
    all_imgs.sort()
    random.seed(42)
    sample_imgs = random.sample(all_imgs, k=min(6, len(all_imgs)))

    prompts = ["turn left", "turn right", "go straight", "slow down", "speed up"]

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    with torch.no_grad():
        text_tokens = tokenizer(prompts)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    scores = []
    for fname in sample_imgs:
        img_path = os.path.join(img_dir, fname)
        image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            img_features = model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            sim = (img_features @ text_features.T).squeeze(0).cpu().numpy().tolist()
        scores.append({"image": fname, "scores": dict(zip(prompts, sim))})

    json_path = os.path.join(out_dir, "vla_clip_scores.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    # Heatmap visualization
    matrix = [[item["scores"][p] for p in prompts] for item in scores]
    plt.figure(figsize=(6, 3))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="cosine similarity")
    plt.xticks(range(len(prompts)), prompts, rotation=30, ha="right")
    plt.yticks(range(len(sample_imgs)), sample_imgs)
    plt.title("CLIP Prompt Scores (VLA Proxy)")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "vla_clip_scores.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("saved_json", json_path)
    print("saved_fig", fig_path)


if __name__ == "__main__":
    main()
