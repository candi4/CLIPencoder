# CLIPencoder
How to use [CLIP](https://openai.com/index/clip/) encoder    

## References
* [CLIP GitHub](https://github.com/openai/CLIP)
* [Interacting with CLIP.ipynb](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb#scrollTo=eFxgLV5HAEEw)


## Installation
1. Installs PyTorch>=1.7.1 and torchvision
    * I installed cuda11.7 and torch
        ```
        conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
        ```
2. Installs dependencies
    ```
    pip install ftfy regex tqdm
    ```
3. Installs [CLIP repo](https://github.com/openai/CLIP) as a Python package
    ```
    pip install git+https://github.com/openai/CLIP.git
    ```
4. Tests installation by running example.py

## Usage
### Getting cosine similarity between image and text
It can encode images and texts in batches and calculate cosine similarities between images and texts.
1. Lists the names of available CLIP models and choose one
    ```
    print(clip.available_models()) # <class 'list'>
    ```
2. Loads CLIP encoder and preprocessor
    ```
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    ```
3. Preprocesses images and tokenizes texts
    ```
    images = [preprocess(Image.open(f"images/maze1.jpeg")),
              preprocess(Image.open(f"images/maze2.jpeg")),
              ]
    image = torch.tensor(np.stack(images)).to(device)
    text = clip.tokenize(['maze', 'game', 'cat', 'win the game']).to(device)
    ```
4. Encodes images and text
    ```
    model.eval()
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    ```
5. Get cosine similarities
    ```
    # Normalizes features
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().detach().numpy()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().detach().numpy()
    
    # Calculates cosine similarities
    similarity = text_features @ image_features.T # similarity[text_i, img_j]
    ```
## Additional Notes
* It takes some time (~0.9s for one image batch) to use the CLIP encoder initially, but subsequent uses are reasonably faster (~0.01s for one image batch).
