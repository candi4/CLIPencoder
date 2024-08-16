# Tests models by comparing the time taken to encode images

import torch
import clip
from PIL import Image
import numpy as np
import time


device = "cuda" if torch.cuda.is_available() else "cpu"

print(clip.available_models())
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
# ViT-B/16
for modelname in clip.available_models():

    print('\n',modelname)
    model, preprocess = clip.load(modelname, device=device)
    
    
    model.eval()
    with torch.no_grad():
        # Use model once with temp image and text
        image_np = np.floor(np.random.random((1,1,3)) *256).astype(np.uint8)
        model.encode_image(preprocess(Image.fromarray(image_np)).unsqueeze(0).to(device))
        model.encode_text(clip.tokenize('0').to(device))

        # Encodes text and normalizes
        text_original = "The lightening point in the maze is reached to the end of the line."
        text = clip.tokenize(text_original).to(device)
        text_features:torch.Tensor = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().detach().numpy()

    for imagename in ['maze1.jpeg','maze2.jpeg']:
        print("- imagename",imagename)
        preprocessstart = time.time()
        image = preprocess(Image.open(f"images/{imagename}")).unsqueeze(0).to(device)
        preprocesstime = time.time() - preprocessstart

        model.eval()
        with torch.no_grad():
            
            encodestart = time.time()
            image_features:torch.Tensor = model.encode_image(image)
            encodetime = time.time() - encodestart
            
            # Encodes image and normalizes
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()
            # Calculates cosine similarity
            similarity_mat = text_features @ image_features.T
            
        print("imagetime", preprocesstime + encodetime)
        assert similarity_mat.size == 1, f"similarity.shape={similarity_mat.shape}"
        similarity = similarity_mat.item()
        print("similarity =",similarity)
