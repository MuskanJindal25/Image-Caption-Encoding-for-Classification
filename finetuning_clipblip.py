import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import matplotlib.pyplot as plt
from FewShotImageFolderNew import FewShotImageFolder
import atexit
from wakepy import keep

# ---------- Paths and Config ----------
DATASET_DIR = "datasets/tiny-imagenet-200"
CLASSNAMES_PATH = "cache/classnames.txt"
os.makedirs("cache", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load Class Names ----------
with open(os.path.join(DATASET_DIR, "words.txt"), 'r') as wf:
    wnid_to_name = dict(line.strip().split('\t') for line in wf.readlines())
#with open(os.path.join(DATASET_DIR, "wnids.txt"), 'r') as wf:
#    wnids = [line.strip() for line in wf.readlines()]

current_directory = os.getcwd()
os.chdir(DATASET_DIR+'/train')
actual_order = sorted(os.listdir())
os.chdir(current_directory)


wnids=actual_order    

classnames = [wnid_to_name[wnid].split(',')[0] for wnid in wnids]
with open(CLASSNAMES_PATH, "w") as f:
    f.write("\n".join(classnames))
print("Class names saved.")

# ---------- Data Loaders ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
NUM_IMAGES_PER_CLASS = 200
train_dir = os.path.join(DATASET_DIR, 'train')
few_shot_train_dataset = FewShotImageFolder(train_dir, wnids=wnids, transform=transform, num_per_class=NUM_IMAGES_PER_CLASS)
val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, 'val'), transform=transform)

train_loader = DataLoader(few_shot_train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ---------- Load Models ----------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.train()

blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.train()  # Enable fine-tuning

# freezing everything but projection
for name, param in clip_model.named_parameters():
    
    if "visual_projection" not in name:
        print(name)
        print("this does not have 'visual_projection' in name")
        param.requires_grad = False
    else: 
        param.requires_grad= True

#freezing blip, just to see

for param in blip_model.vision_model.parameters():
    param.requires_grad = False

# ---------- Optimizer ----------
optimizer = torch.optim.AdamW(
    list(clip_model.parameters()) + list(blip_model.parameters()),
    lr=5e-5
)

# ---------- ICE Fine-tuning Loop ----------
lambda_weight = 0.05
EPOCHS = 3
temperature = 0.01
losses = []

with torch.no_grad():
    all_class_prompts = [f"a photo of a {name}" for name in classnames]
    class_inputs = clip_processor(text=all_class_prompts, return_tensors="pt", padding=True).to(device)
    full_class_embs = clip_model.get_text_features(**class_inputs)
    full_class_embs = full_class_embs / full_class_embs.norm(dim=-1, keepdim=True)

num_classes = len(classnames)
with keep.running():
    for epoch in range(EPOCHS):
        print(f"\n Epoch {epoch + 1}/{EPOCHS} - ICE Fine-tuning")
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="ICE Training"):
            images = images.to(device)
    
            # ---- Generate BLIP captions ----
            #pseudo_captions = [f"a photo of a {classnames[label.item()]}" for label in labels]
            # added caption conditioning
            #texts = [f"a photo of a {classnames[label]}" for label in labels]
            texts = [f"a photo of a" for label in labels]
            blip_inputs = blip_processor(
                images=images,
                #text=["a photo"] * len(images),
                text=texts,
                return_tensors="pt",
                padding=True,
                do_rescale=False,
                padding_side='left'
            ).to(device)
    
            generated_ids = blip_model.generate(**blip_inputs)
            captions = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
    
            # ---- Get CLIP image embeddings ----
            clip_inputs = clip_processor(images=images, return_tensors="pt", do_rescale=False).to(device)
            image_embs = clip_model.get_image_features(**clip_inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
    
            # ---- Get CLIP caption embeddings ----
            text_inputs = clip_processor(text=captions, return_tensors="pt", padding=True).to(device)
            caption_embs = clip_model.get_text_features(**text_inputs)
            caption_embs = caption_embs / caption_embs.norm(dim=-1, keepdim=True)
    
            # ---- ICE embeddings ----
            ice_embs = (1 - lambda_weight) * image_embs + lambda_weight * caption_embs
            ice_embs = ice_embs / ice_embs.norm(dim=-1, keepdim=True)
    
            # ---- Class prompt embeddings ----
            #class_prompts = [f"a photo of a {classnames[label]}" for label in labels]
            #class_inputs = clip_processor(text=class_prompts, return_tensors="pt", padding=True).to(device)
            #class_embs = clip_model.get_text_features(**class_inputs)
            #class_embs = class_embs / class_embs.norm(dim=-1, keepdim=True)
    
            # ---- Compute Loss ----
            if labels.max() >= num_classes or labels.min() < 0:
                raise ValueError(f"Label out of bounds. Got min={labels.min()}, max={labels.max()}, expected in [0, {num_classes - 1}]")
            logits = torch.matmul(ice_embs, full_class_embs.T) / temperature
            target = labels.to(device)
            loss = torch.nn.functional.cross_entropy(logits, target, label_smoothing = 0.1)
    
            # ---- Backward ----
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
            losses.append(loss.item())
            running_loss += loss.item()
    
        print(f"Epoch {epoch + 1} Avg Loss: {running_loss / len(train_loader):.4f}")

    # ---------- Evaluation on Validation Set ----------
    '''clip_model.eval()
    blip_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # BLIP caption generation
            blip_inputs = blip_processor(images=images, text=["a photo"] * len(images), return_tensors="pt", padding=True, do_rescale=False).to(device)
            generated_ids = blip_model.generate(**blip_inputs)
            captions = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)

            # ICE embedding
            clip_inputs = clip_processor(images=images, return_tensors="pt", do_rescale=False).to(device)
            image_embs = clip_model.get_image_features(**clip_inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

            text_inputs = clip_processor(text=captions, return_tensors="pt", padding=True).to(device)
            caption_embs = clip_model.get_text_features(**text_inputs)
            caption_embs = caption_embs / caption_embs.norm(dim=-1, keepdim=True)

            ice_embs = (1 - lambda_weight) * image_embs + lambda_weight * caption_embs
            ice_embs = ice_embs / ice_embs.norm(dim=-1, keepdim=True)

            # Class embeddings
            class_prompts = [f"a photo of a {name}" for name in classnames]
            class_inputs = clip_processor(text=class_prompts, return_tensors="pt", padding=True).to(device)
            class_embs = clip_model.get_text_features(**class_inputs)
            class_embs = class_embs / class_embs.norm(dim=-1, keepdim=True)

            logits = torch.matmul(ice_embs, class_embs.T)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Validation Accuracy after Epoch {epoch + 1}: {acc:.2f}%")

    clip_model.train()
    blip_model.train()'''

# ---------- Save Models ----------
torch.save(clip_model.state_dict(), "models/reg_ice_clip.pth")
torch.save(blip_model.state_dict(), "models/reg_ice_blip.pth")
print("Fine-tuned models saved to 'models/'.")

# ---------- Plot Training Loss ----------
plt.figure(figsize=(10, 4))
plt.plot(losses, label="ICE Training Loss", color="blue")
plt.xlabel("Iteration")
plt.ylabel("Cross-Entropy Loss")
plt.title("ICE Training Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/loss_curve.png")
print("Loss plot saved to 'plots/loss_curve.png'")

@atexit.register
def cleanup():
    print("Cleaning up...")
    torch.cuda.empty_cache()