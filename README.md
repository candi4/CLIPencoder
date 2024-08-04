# CLIPencoder
How to use [CLIP](https://openai.com/index/clip/) encoder    

## References
* [CLIP GitHub](https://github.com/openai/CLIP)
* [Interacting with CLIP.ipynb](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb#scrollTo=eFxgLV5HAEEw)
* [VLM-RM GitHub](https://github.com/AlignmentResearch/vlmrm)
* [J. Rocamonde, V. Montesinos, E. Nava, E. Perez, and D. Lindner. Vision-language models are zero-shot re-ward models for reinforcement learning. arXiv preprint arXiv:2310.12921, 2023.](https://openreview.net/forum?id=N0I2RtD8je)


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
    clip.available_models() # <class 'list'>
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



### Goal-baseline regularization ([VLM-RM](https://sites.google.com/view/vlm-rm))
In VLM-RM, they regularized cosine similarity by adding baseline description.

$$R_{\text{CLIP-Reg}}(s) = 1 - \frac{1}{2} \lVert \alpha \text{ proj}_{L} \mathbf{s} + (1 - \alpha) \mathbf{s} - \mathbf{g} \rVert_2^2$$
* $\mathbf{s}$ is the normalized vector obtained by encoding the image observation.
* $\mathbf{g}$ is the normalized vector obtained by encoding the goal task description.    
* $\mathbf{b}$ is the normalized vector obtained by encoding the baseline description.    
* $L$ is the line passing the points $\mathbf{g}$ and $\mathbf{b}$.
* $\alpha$ is a parameter to control the regularization strength.
1. Load CLIP encoder and preprocessor
    ```
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    ```
2. Encode and normalize to produce $\mathbf{s}$, $\mathbf{g}$, and $\mathbf{b}$.
    ```
    # s
    image_observations = [preprocess(Image.open(f"images/maze1.jpeg")),
                          preprocess(Image.open(f"images/maze2.jpeg")),
                          ]
    image_observation = torch.tensor(np.stack(image_observations)).to(device)
    state:torch.Tensor = model.encode_image(image_observation)
    state /= state.norm(dim=-1, keepdim=True)

    # g
    goal_text = clip.tokenize(['win the game', 'navigate to the goal']).to(device)
    target:torch.Tensor = model.encode_text(goal_text)
    target /= target.norm(dim=-1, keepdim=True)
    target = target.mean(dim=0, keepdim=True)
    
    # b
    baseline_text = clip.tokenize(['maze', 'game', 'navigation']).to(device)
    baseline:torch.Tensor = model.encode_text(baseline_text)
    baseline /= baseline.norm(dim=-1, keepdim=True)
    baseline = baseline.mean(dim=0, keepdim=True)
    ```
3. Calculate reward
    ```
    # R
    alpha = 0.5
    direction = target - baseline
    projection = direction.T @ direction / torch.norm(direction) ** 2
    identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
    projection = alpha * projection + (1 - alpha) * identity
    rewards = 1 - (torch.norm((state - target) @ projection, dim=-1) ** 2) / 2
    ```
## VLM-RM




## Additional Notes
* It takes some time (~0.9s for one image batch) to use the CLIP encoder initially, but subsequent uses are reasonably faster (~0.01s for one image batch).
