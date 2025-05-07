import os
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import atexit
import matplotlib.pyplot as plt
from torch.amp import autocast
import json


# Load fine-tuned models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.load_state_dict(torch.load("models/reg_ice_clip.pth"))
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.load_state_dict(torch.load("models/reg_ice_blip.pth"))
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

clip_model.eval()
blip_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
blip_model.to(device)


# -----------  Load Class Names from words.txt and wnids.txt -----------
DATASET_DIR = "datasets/tiny-imagenet-200"
with open(os.path.join(DATASET_DIR, "words.txt"), 'r') as wf:
    wnid_to_name = dict(line.strip().split('\t') for line in wf.readlines())
#with open(os.path.join(DATASET_DIR, "wnids.txt"), 'r') as wf:
#    orig_wnids = [line.strip() for line in wf.readlines()]

# DATASET_DIR = "/projectnb/ec523/projects/proj_ICE/ImageNet_subset"
# with open(f"{DATASET_DIR}/Labels.json", 'r') as jf:
#     wnid_to_name = json.load(jf)


# DATASET_DIR = "datasets/tiny-imagenet-200"
# CLASSNAMES_PATH = "cache/classnames.txt"
current_directory = os.getcwd()

# os.chdir(DATASET_DIR+'/val')
# actual_order = sorted(os.listdir())

val_dir = "/projectnb/ec523/projects/proj_ICE/ImageNet_subset/train.X4"
actual_order = sorted(os.listdir(val_dir))

os.chdir(current_directory)

wnids=actual_order    
classnames = [wnid_to_name[wnid].split(',')[0] for wnid in wnids]



# Create prompts
prompt_texts = [f"a photo of a {name}" for name in classnames]
label_inputs = clip_processor(text=prompt_texts, return_tensors="pt", padding=True).to(device)
label_embeds = clip_model.get_text_features(**label_inputs)
label_embeds = label_embeds / label_embeds.norm(dim=-1, keepdim=True)


# -----------  Validation DataLoader -----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=transform)
val_dir = "/projectnb/ec523/projects/proj_ICE/ImageNet_subset/train.X4"
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

val_dataset.class_to_idx = {wnid: i for i, wnid in enumerate(actual_order)}
val_dataset.classes = actual_order  # explicitly set the order


val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ----------- ICE Inference -----------
print("\n Running Batched ICE Inference with Top-k Caption Selection")
total = 0
correct = 0
#lambda_weight = 0.05
#lambda_weight = 0.01
lambda_weight = 0.05
k = 3  # number of captions per image
temperature = 0.1

#hell_test(val_loader, val_dataset)


for batch_idx, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader), desc="ICE Inference"):
    images = images.to(device)
    labels = labels.to(device)
    batch_size = images.size(0)
    # Display the first image in the current batch: only uncomment if there's a similar problem to earlier
    #first_img = images[0].cpu().permute(1, 2, 0).numpy()  # CHW to HWC
    #plt.imshow(first_img)
    #plt.title("First Image in Batch")
    #plt.axis('off')
    #plt.show()

    with torch.no_grad(), autocast('cuda'):
        # Repeat images k times
        repeated_images = images.unsqueeze(1).repeat(1, k, 1, 1, 1).view(-1, *images.shape[1:])  # [B*k, C, H, W]

        # Generate captions
        blip_inputs = blip_processor(
            images=repeated_images,
            text=["a photo of"] * (batch_size * k),
            return_tensors="pt",
            padding=True,
            do_rescale=False
        ).to(device)

        generated_ids = blip_model.generate(**blip_inputs, max_new_tokens=20, do_sample=True, top_p=0.9, temperature=1.0)
        captions = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Caption embeddings
        text_inputs = clip_processor(text=captions, return_tensors="pt", padding=True).to(device)
        caption_embeds = clip_model.get_text_features(**text_inputs)
        caption_embeds = caption_embeds / caption_embeds.norm(dim=-1, keepdim=True)

        # Image embeddings
        clip_inputs = clip_processor(images=repeated_images, return_tensors="pt", do_rescale=False).to(device)
        image_embeds = clip_model.get_image_features(**clip_inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # ICE fusion
        ice_embeds = (1 - lambda_weight) * image_embeds + lambda_weight * caption_embeds
        ice_embeds = ice_embeds / ice_embeds.norm(dim=-1, keepdim=True)

        # Class similarity
        sims = torch.matmul(ice_embeds, label_embeds.T)   # [B*k, C]
        sims_per_image = sims.view(batch_size, k, -1)    # [B, k, C]

        # Best caption per image
        best_indices = (sims_per_image/temperature).max(dim=2).values.argmax(dim=1)  # [B]
        final_ice_embeds = []
        selected_captions = []

        for i in range(batch_size):
            idx = i * k + best_indices[i].item()
            #final_ice_embeds.append(ice_embeds[idx])
            selected_captions.append(captions[idx])

        #final_ice_embeds = torch.stack(final_ice_embeds)
        #trying out multi-caption averaging
        
        ice_embeds_reshaped = ice_embeds.view(batch_size, k, -1)  # [B, k, D]
        weights = (sims_per_image/temperature).max(dim=2).values.softmax(dim=1)  # [B, k] — softmax over best score per caption
        final_ice_embeds = torch.sum(weights.unsqueeze(-1) * ice_embeds_reshaped, dim=1)  # [B, D]
        final_ice_embeds = final_ice_embeds / final_ice_embeds.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(final_ice_embeds, label_embeds.T) 
        preds = similarity.argmax(dim=1)
        

        total += labels.size(0)
        correct += (preds.cpu() == labels.cpu()).sum().item()

    accuracy = 100.0 * correct / total
    print(f"\n Accuracy so far: {accuracy:.2f}%")
    for i, caption in enumerate(selected_captions):
        pred_idx = preds[i].item()
        true_idx = labels[i].item()
        print(
            f" Image {i+1}: Caption: \"{caption}\"\n"
            f"    Predicted: {classnames[pred_idx]} ({wnids[pred_idx]})\n"
            f"    Actual:    {classnames[true_idx]} ({wnids[true_idx]})\n"
        )

print(f"\n ICE Inference Complete | Final Accuracy: {accuracy:.2f}%")

@atexit.register
def cleanup():
    print("Cleaning up...")
    torch.cuda.empty_cache()


