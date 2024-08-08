from typing import Union

import numpy as np
from PIL import Image
import time

import torch as th
import torch.nn as nn
import clip

from VLMRM.utils import get_device

class CLIPEmbed(nn.Module):
    def __init__(self, clip_model_name, device:Union[th.device, str]='auto',):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.device = device
        self.model, self.preprocess = clip.load(self.clip_model_name, device=self.device)
        
    @th.inference_mode()
    def embed_images(self, images:list, normalize=True) -> th.Tensor:
        assert isinstance(images[0], Image.Image), f'Given type: {type(images[0])}'
        images = [self.preprocess(image) for image in images]
        images = th.tensor(np.stack(images)).to(self.device)
        vector:th.Tensor = self.model.encode_image(images).float()
        if normalize:
            vector /= vector.norm(dim=-1, keepdim=True)
        return vector
    
    @th.inference_mode()
    def embed_texts(self, texts:list, normalize=True, mean=True) -> th.Tensor:
        assert isinstance(texts[0], str), f'Given type: {type(texts[0])}'
        texts = clip.tokenize(texts).to(self.device)
        vector:th.Tensor = self.model.encode_text(texts).float()
        if normalize:
            vector /= vector.norm(dim=-1, keepdim=True)
        if mean:
            vector = vector.mean(dim=0, keepdim=True)
        return vector
        
    

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
        
        self.target = None
        self.baseline = None
        self.projection = None

        self.model, self.preprocess = clip.load(self.clip_model_name, device=self.device)
        self.clip_embed = CLIPEmbed(clip_model_name=self.clip_model_name, device=self.device)
        
    def set(self, alpha:float,
            target_raw=None,
            *,
            baseline_raw=None,
            ):
        if target_raw is not None:
            self.target = self.clip_embed.embed_texts(texts=target_raw, normalize=True, mean=True)
        if baseline_raw is not None:
            self.baseline = self.clip_embed.embed_texts(texts=baseline_raw, normalize=True, mean=True)
        
        assert not ((alpha != 0) and (self.baseline is None)), f"alpha={alpha}, self.baseline={self.baseline}"    
            
        if alpha == 0:
            identity = th.diag(th.ones(projection.shape[0])).to(projection.device)
            self.projection = identity
        else:
            direction = self.target - self.baseline
            projection = direction.T @ direction / th.norm(direction) ** 2
            identity = th.diag(th.ones(projection.shape[0])).to(projection.device)
            self.projection = alpha * projection + (1 - alpha) * identity

    @th.inference_mode()
    def get_rewards(self, image_observations: list) -> th.Tensor:
        """
        :param image_observations: list of PIL.Image.Image
        """
        state = self.clip_embed.embed_images(images=image_observations, normalize=True)
        
        reward = 1 - (th.norm((state - self.target) @ self.projection, dim=-1) ** 2) / 2
        return reward