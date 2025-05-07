# Image-Caption-Encoding-for-Classification
Using a CLIP model and various captioner's for image caption encoding for the goal of zero-shot image classification


How to run:
Zero‑Shot Caption Fusion with CLIP

This project provides two Python scripts for zero‑shot image classification using OpenAI’s CLIP model, enhanced with automatically generated image captions:

1. Ensemble_clip_eval.py
   - Runs CLIP in zero‑shot mode using a diverse set of text prompts.  
   - For each class, generates 20 prompt templates (e.g. “a photo of a {class}”) and average the resulting CLIP softmax scores.

2. **zero_shot_caption_fusion.py**  
   - Produces image captions using BLIP, GIT, and ViT‑GPT2 models.  
   - Computes CLIP embeddings for both the image and its caption, then fuses them via   
     v_fused = α * v_image + (1 - α) * v_text  
   - Re‑ranks the top K image-only candidates and selects the highest‑scoring label.


## Requirements

- Python 3.8 or later  
- CUDA 11.7 or later (for GPU acceleration)  
- PyTorch 2.0.1  
- Transformers 4.28.1  
- OpenAI CLIP package  
- SentencePiece (required by some captioners)  
- Pillow, NumPy, Matplotlib  


