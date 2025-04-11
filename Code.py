import os
import csv
import json
from PIL import Image
import torch
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
clip_model, preprocess = clip.load("ViT-B/32", device=device)

print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

print("Loading Labels.json...")
with open("/projectnb/ec523/projects/proj_ICE/ImageNet_subset/Labels.json", "r") as f:
    true_labels = json.load(f)

val_folder = "ImageNet_subset/val.X"
output_file = "caption_vs_classification_results_fixed.csv"

# Build candidate class list from label names (take only the first keyword for each class)
candidate_labels = sorted(set(label.lower().split(",")[0] for label in true_labels.values()))

def get_blip_caption(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def run_clip(image, options):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(options).to(device)
    with torch.no_grad():
        logits_per_image, _ = clip_model(image_input, text_inputs)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probs, options[probs.argmax()]

total_images = 0
blip_correct = 0
clip_correct = 0
rows = []

print("Starting evaluation...")

for folder in os.listdir(val_folder):
    class_path = os.path.join(val_folder, folder)
    true_label = true_labels.get(folder, "").lower()
    true_keywords = [word.strip() for word in true_label.split(",")]

    if not os.path.isdir(class_path):
        continue

    print(f"Processing folder: {folder}...")

    for filename in os.listdir(class_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        full_path = os.path.join(class_path, filename)
        try:
            img = Image.open(full_path).convert("RGB")
            caption = get_blip_caption(img)

            # Run CLIP using BLIP caption + label candidates
            all_options = [caption] + candidate_labels
            probs, clip_guess = run_clip(img, all_options)

            # Check if true keywords appear in the predictions
            blip_match = any(keyword in caption.lower() for keyword in true_keywords)
            clip_match = any(keyword in clip_guess.lower() for keyword in true_keywords)

            rows.append([filename, folder, caption, clip_guess, true_label, blip_match, clip_match])
            total_images += 1
            blip_correct += int(blip_match)
            clip_correct += int(clip_match)

            if total_images % 5 == 0:
                print(f"Processed {total_images} images...")

        except Exception as e:
            print(f"�~]~L Skipped {filename}: {e}")

print("Writing results to CSV...")

with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "folder", "blip_caption", "clip_prediction", "true_label", "blip_correct", "clip_correct"])
    writer.writerows(rows)
    writer.writerow([])
    writer.writerow(["TOTALS", "", "", "", "", blip_correct, clip_correct])
    writer.writerow(["ACCURACY", "", "", "", "", f"{(blip_correct/total_images):.2%}", f"{(clip_correct/total_images):.2%}"])

print("�~\~E Done! Results written to caption_vs_classification_results_fixed.csv")
