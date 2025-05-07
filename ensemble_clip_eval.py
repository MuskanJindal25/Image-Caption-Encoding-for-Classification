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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "a vintage photo of a {}",
]

def get_blip_caption(img):
    x = blip_proc(img, return_tensors="pt").to(device)
    return blip_proc.decode(
        blip_model.generate(**x)[0], skip_special_tokens=True
    )

def get_git_caption(img):
    x = git_proc(images=img, return_tensors="pt").to(device)
    return git_proc.batch_decode(
        git_model.generate(**x), skip_special_tokens=True
    )[0]

def get_v2_caption(img):
    fe = vitgpt2_feat(images=img, return_tensors="pt")
    pixel_vals = fe["pixel_values"].to(device)
    return vitgpt2_tok.batch_decode(
        vitgpt2_model.generate(pixel_vals), skip_special_tokens=True
    )[0]

@torch.no_grad()
def run_clip_with_ensemble(classes, img):
    texts = [t.format(cls) for cls in classes for t in ensemble_templates]
    class_idx = [cls for cls in classes for _ in ensemble_templates]

    ii = preprocess(img).unsqueeze(0).to(device)
    ti = clip.tokenize(texts).to(device)
    logits, _ = clip_model(ii, ti)
    probs = logits.softmax(dim=-1)[0].cpu().tolist()

    sums = {cls: 0.0 for cls in classes}
    counts = {cls: 0 for cls in classes}
    for p, cls in zip(probs, class_idx):
        sums[cls] += p
        counts[cls] += 1

    avg_probs = {cls: sums[cls] / counts[cls] for cls in classes}
    return max(avg_probs, key=avg_probs.get)

@torch.no_grad()
def fuse_and_classify_topk(img, cap, classes, alpha=0.5, K=5, kws=None):
    ii = preprocess(img).unsqueeze(0).to(device)
    ifeat = clip_model.encode_image(ii)
    tfeat = clip_model.encode_text(clip.tokenize([cap]).to(device))
    cf_all = clip.tokenize(classes).to(device)
    cfeat_all = clip_model.encode_text(cf_all)
    sims_img = (ifeat @ cfeat_all.T).softmax(-1)[0]
    topk = sims_img.topk(K).indices.tolist()
    cand_classes = [classes[i] for i in topk]
    cand_feats = cfeat_all[topk]
    fused = torch.nn.functional.normalize(alpha * ifeat + (1 - alpha) * tfeat, dim=-1)
    sims_fuse = (fused @ cand_feats.T).softmax(-1)[0]
    return cand_classes[sims_fuse.argmax().item()]

with open("ImageNet_subset/Labels.json") as f:
    true_labels = json.load(f)
classes = sorted({v.split(",")[0].lower() for v in true_labels.values()})

BEST_ALPHA = {"blip": 0.8, "git": 0.8, "v2": 0.8}
BEST_K     = {"blip":   5, "git":   5, "v2":   5}
max_images = 5000

out = "testingrightnow.csv"
w = csv.writer(open(out, "w", newline=""))
w.writerow([
    "image","folder","true_label",
    "blip_caption","git_caption","vitgpt2_caption",
    "clip_only","clip_blip","clip_git","clip_v2",
    "fuse_blip","fuse_git","fuse_v2",
    "blip_ok","git_ok","v2_ok",
    "clip_ok","clip_blip_ok","clip_git_ok","clip_v2_ok",
    "fuse_blip_ok","fuse_git_ok","fuse_v2_ok"
])

count = 0
totals = {k: 0 for k in [
    "blip","git","v2","clip","clip_blip","clip_git","clip_v2",
    "fuse_blip","fuse_git","fuse_v2"
]}

for fld in sorted(os.listdir("ImageNet_subset/train.X1")):
    path = os.path.join("ImageNet_subset/train.X1", fld)
    if not os.path.isdir(path): continue
    kws = [w.strip() for w in true_labels[fld].lower().split(",")]

    for imgf in sorted(os.listdir(path)):
        if count >= max_images: break
        if not imgf.lower().endswith((".jpg",".jpeg",".png")): continue

        img = Image.open(os.path.join(path, imgf)).convert("RGB")
        b = get_blip_caption(img)
        g = get_git_caption(img)
        v = get_v2_caption(img)

        c0 = run_clip_with_ensemble(classes, img)
        c1 = run_clip_with_ensemble([b] + classes, img)
        c2 = run_clip_with_ensemble([g] + classes, img)
        c3 = run_clip_with_ensemble([v] + classes, img)

        f1 = fuse_and_classify_topk(img, b, classes,
                                    alpha=BEST_ALPHA["blip"],
                                    K=BEST_K["blip"],
                                    kws=kws)
        f2 = fuse_and_classify_topk(img, g, classes,
                                    alpha=BEST_ALPHA["git"],
                                    K=BEST_K["git"],
                                    kws=kws)
        f3 = fuse_and_classify_topk(img, v, classes,
                                    alpha=BEST_ALPHA["v2"],
                                    K=BEST_K["v2"],
                                    kws=kws)

        ok = lambda p: any(k in p for k in kws)
        flags = {
            "blip": ok(b.lower()), "git": ok(g.lower()), "v2": ok(v.lower()),
            "clip": ok(c0), "clip_blip": ok(c1), "clip_git": ok(c2), "clip_v2": ok(c3),
            "fuse_blip": ok(f1), "fuse_git": ok(f2), "fuse_v2": ok(f3)
        }
        for k in totals:
            totals[k] += int(flags[k])
        count += 1

        w.writerow([
            imgf, fld, true_labels[fld].lower(),
            b, g, v,
            c0, c1, c2, c3,
            f1, f2, f3,
            flags["blip"], flags["git"], flags["v2"],
            flags["clip"], flags["clip_blip"], flags["clip_git"], flags["clip_v2"],
            flags["fuse_blip"], flags["fuse_git"], flags["fuse_v2"]
        ])
    if count >= max_images: break

w.writerow([])
w.writerow(["ACCURACY","",""] + [
    f"{totals[k]/count:.2%}" for k in [
        "blip","git","v2","clip","clip_blip","clip_git","clip_v2",
        "fuse_blip","fuse_git","fuse_v2"
    ]
])
print("Done", count, "images", out)

