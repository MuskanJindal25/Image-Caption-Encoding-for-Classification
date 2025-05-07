import os
import csv
import json
from PIL import Image
import torch
import clip
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq,
    AutoFeatureExtractor, AutoTokenizer
)
import numpy as np
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

git_proc = AutoProcessor.from_pretrained("microsoft/git-base")
git_model = AutoModelForVision2Seq.from_pretrained("microsoft/git-base").to(device)

vitgpt2_feat = AutoFeatureExtractor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
vitgpt2_model = AutoModelForVision2Seq.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
).to(device)
vitgpt2_tok = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

ensemble_templates = [
    "a photo of a {}", "a picture of a {}", "a rendering of a {}", "a cropped photo of a {}",
    "a bright photo of a {}", "a zoomed-in photo of a {}", "a close-up of the {}",
    "a shot of a {} in the wild", "a low-resolution photo of a {}", "a high-resolution photo of a {}",
    "an artful depiction of a {}", "a DSLR shot of a {}", "a blurry photo of a {}",
    "a colorful painting of a {}", "an overhead shot of a {}", "a fast-moving {}",
    "a detailed image of a {}", "a minimalistic photo of a {}", "a studio photo of a {}",
    "a vintage photo of a {}"
]

def get_blip_caption(img):
    x = blip_proc(img, return_tensors="pt").to(device)
    return blip_proc.decode(
        blip_model.generate(**x)[0], skip_special_tokens=True
    ).lower()

def get_git_caption(img):
    x = git_proc(images=img, return_tensors="pt").to(device)
    return git_proc.batch_decode(
        git_model.generate(**x), skip_special_tokens=True
    )[0].lower()

def get_v2_caption(img):
    fe = vitgpt2_feat(images=img, return_tensors="pt")
    pixel_vals = fe["pixel_values"].to(device)
    return vitgpt2_tok.batch_decode(
        vitgpt2_model.generate(pixel_vals), skip_special_tokens=True
    )[0].lower()

@torch.no_grad()
def run_clip_with_ensemble(classes, img):
    texts = [t.format(cls) for cls in classes for t in ensemble_templates]
    idx = [cls for cls in classes for _ in ensemble_templates]
    ii = preprocess(img).unsqueeze(0).to(device)
    ti = clip.tokenize(texts).to(device)
    logits, _ = clip_model(ii, ti)
    probs = logits.softmax(dim=-1)[0].cpu().tolist()
    sums = {cls: 0.0 for cls in classes}
    counts = {cls: 0 for cls in classes}
    for p, c in zip(probs, idx):
        sums[c] += p; counts[c] += 1
    avg = {c: sums[c]/counts[c] for c in classes}
    return max(avg, key=avg.get)

@torch.no_grad()
def fuse_and_classify_topk(img, cap, classes, alpha=0.5, K=5):
    ii = preprocess(img).unsqueeze(0).to(device)
    ifeat = clip_model.encode_image(ii)
    tfeat = clip_model.encode_text(clip.tokenize([cap]).to(device))
    cf = clip.tokenize(classes).to(device)
    cfeat = clip_model.encode_text(cf)
    sims = (ifeat @ cfeat.T).softmax(-1)[0]
    topk = sims.topk(K).indices.tolist()
    cand = [classes[i] for i in topk]
    cand_feats = cfeat[topk]
    fused = torch.nn.functional.normalize(alpha*ifeat + (1-alpha)*tfeat, dim=-1)
    sims_f = (fused @ cand_feats.T).softmax(-1)[0]
    return cand[sims_f.argmax().item()]

with open("ImageNet_subset/Labels.json") as f:
    true_labels = json.load(f)
classes = sorted({v.split(',')[0].lower() for v in true_labels.values()})

max_images = 5000
out = "results_by_method.csv"
totals = defaultdict(int)
count = 0

with open(out, 'w', newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow([
        "image","folder","true_label",
        "blip_cap","git_cap","v2_cap",
        "clip_only","clip_blip","clip_git","clip_v2",
        "fus_blip","fus_git","fus_v2"
    ])

    for fld in sorted(os.listdir("ImageNet_subset/train.X1")):
        path = os.path.join("ImageNet_subset/train.X1", fld)
        if not os.path.isdir(path): continue
        kws = [k.strip() for k in true_labels[fld].split(',')]

        for imgf in sorted(os.listdir(path)):
            if count >= max_images: break
            if not imgf.lower().endswith(('jpg','jpeg','png')): continue
            img = Image.open(os.path.join(path, imgf)).convert('RGB')

            b = get_blip_caption(img)
            g = get_git_caption(img)
            v = get_v2_caption(img)

            c0 = run_clip_with_ensemble(classes, img)
            c1 = run_clip_with_ensemble([b]+classes, img)
            c2 = run_clip_with_ensemble([g]+classes, img)
            c3 = run_clip_with_ensemble([v]+classes, img)

            f1 = fuse_and_classify_topk(img, b, classes, alpha=0.2, K=3)
            f2 = fuse_and_classify_topk(img, g, classes, alpha=0.2, K=3)
            f3 = fuse_and_classify_topk(img, v, classes, alpha=0.2, K=3)

            row = [imgf, fld, true_labels[fld].split(',')[0].lower(),
                   b, g, v, c0, c1, c2, c3, f1, f2, f3]
            writer.writerow(row)

            for name, pred in zip(
                ['blip','git','v2','clip','clip_blip','clip_git','clip_v2','fus_blip','fus_git','fus_v2'],
                [b,g,v,c0,c1,c2,c3,f1,f2,f3]
            ):
                if any(k in pred for k in kws): totals[name]+=1
            count +=1
        if count>=max_images: break

    writer.writerow([])
    writer.writerow(["ACCURACY","",""] +
                    [f"{totals[m]/count:.2%}" for m in
                    ['blip','git','v2','clip','clip_blip','clip_git','clip_v2','fus_blip','fus_git','fus_v2']])

print(f"Done {count} images. Results in {out}")

