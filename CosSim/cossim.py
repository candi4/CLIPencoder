# General code for getting cosine similarity using CLIP encoder

import torch
import clip
from PIL import Image
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

# Loads clip encoder
model, preprocess = clip.load('ViT-B/16', device=device)

# Preprocesses images and tokenizes texts
images = [preprocess(Image.open(f"images/maze1.jpeg")),
          preprocess(Image.open(f"images/maze2.jpeg")),
          ]
image = torch.tensor(np.stack(images)).to(device)
text = clip.tokenize(['maze', 'game', 'cat', 'win the game']).to(device)

model.eval()
with torch.no_grad():
    # Encodes images and texts
    image_features:torch.Tensor = model.encode_image(image)
    text_features:torch.Tensor = model.encode_text(text)
    
    # Normalizes features
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().detach().numpy()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    
    # Calculates cosine similarities
    similarity = text_features @ image_features.T # similarity[text_i, img_j]
    
print(similarity)
