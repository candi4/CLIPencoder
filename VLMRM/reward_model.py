from typing import Union

import numpy as np
from PIL import Image
import time

import torch as th
import torch.nn as nn
import clip

from VLMRM.utils import get_device

class CLIPEmbed(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        if isinstance(clip_model.visual.image_size, int):
            image_size = clip_model.visual.image_size
        else:
            image_size = clip_model.visual.image_size[0]
        self.transform = image_transform(image_size)

    @th.inference_mode()
    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        with th.no_grad(), th.autocast("cuda", enabled=th.cuda.is_available()):
            x = self.transform(x)
            x = self.clip_model.encode_image(x, normalize=True)
        return x

class CLIPReward(nn.Module):
    def __init__(self, clip_model_name:str, 
                 *, 
                 is_state_image=True, is_target_image=False, is_baseline_image=False,
                 device:Union[th.device, str]='auto',
                 ):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.is_state_image = is_state_image
        self.is_target_image = is_target_image
        self.is_baseline_image = is_baseline_image
        self.device = get_device(device)

        self.model, self.preprocess = clip.load(self.clip_model_name, device=self.device)
    def set(self, alpha,
            target=None,
            *,
            baseline=None,
            ):
        
        # g
        goal_text = clip.tokenize(['win the game', 'navigate to the goal']).to(self.device)
        target:th.Tensor = self.model.encode_text(goal_text)
        target /= target.norm(dim=-1, keepdim=True)
        target = target.mean(dim=0, keepdim=True)
        self.target = target
        
        # b
        baseline_text = clip.tokenize(['maze', 'game', 'navigation']).to(self.device)
        baseline:th.Tensor = self.model.encode_text(baseline_text)
        baseline /= baseline.norm(dim=-1, keepdim=True)
        baseline = baseline.mean(dim=0, keepdim=True)
        self.baseline = baseline
        
        alpha = 0.5
        direction = self.target - self.baseline
        projection = direction.T @ direction / th.norm(direction) ** 2
        identity = th.diag(th.ones(projection.shape[0])).to(projection.device)
        self.projection = alpha * projection + (1 - alpha) * identity


    @th.inference_mode()
    def get_rewards(self, image_observations: list) -> th.Tensor:
        # s
        with th.no_grad():
            s = time.time()
            image_observations = [self.preprocess(image) for image in image_observations]
            image_observation = th.tensor(np.stack(image_observations)).to(self.device)
            state:th.Tensor = self.model.encode_image(image_observation)
            state /= state.norm(dim=-1, keepdim=True)

            reward = 1 - (th.norm((state - self.target) @ self.projection, dim=-1) ** 2) / 2
            print(time.time()-s)
            return reward