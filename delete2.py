import torch
import clip
from PIL import Image
import numpy as np

import time


device = "cuda" if torch.cuda.is_available() else "cpu"

# Loads clip encoder
model, preprocess = clip.load('ViT-B/16', device=device)

model.eval()
with torch.no_grad():


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

#
# (2, 512)
# torch.Size([2, 512])
# (1, 512)
# torch.Size([3, 512])
# (1, 512)

    alpha = 0.5
    direction = target - baseline
    projection = direction.T @ direction / torch.norm(direction) ** 2
    identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
    projection = alpha * projection + (1 - alpha) * identity
    for i in range(10):
        s = time.time()
        # s
        image_observations = [preprocess(Image.open(f"Usage/images/maze1.jpeg")),
                              preprocess(Image.open(f"Usage/images/maze2.jpeg")),
                              ]
        image_observation = torch.tensor(np.stack(image_observations)).to(device)
        state:torch.Tensor = model.encode_image(image_observation)
        state /= state.norm(dim=-1, keepdim=True)

        # R
        rewards = 1 - (torch.norm((state - target) @ projection, dim=-1) ** 2) / 2
        print(time.time()-s)

    print(rewards)
print(rewards[0].item())