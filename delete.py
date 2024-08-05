from VLMRM import CLIPReward

from PIL import Image
import clip

print(type(CLIPReward))


# model, preprocess = clip.load('ViT-B/16', device=device)

vlmrm = CLIPReward(clip_model_name='ViT-B/16',
            is_state_image=True, is_target_image=False, is_baseline_image=False)

vlmrm.set(target=['win the game', 'navigate to the goal'],
        baseline=['maze', 'game', 'navigation'],
        alpha=0.5
        )
for i in range(10):
    rewards = vlmrm.get_rewards(image_observations=[Image.open(f"Usage/images/maze1.jpeg"),
                                                    Image.open(f"Usage/images/maze2.jpeg"),
                                                    ])

    print(rewards)

#     # s
#     image_observations = [preprocess(Image.open(f"images/maze1.jpeg")),
#                           preprocess(Image.open(f"images/maze2.jpeg")),
#                           ]
#     image_observation = torch.tensor(np.stack(image_observations)).to(device)
#     state:torch.Tensor = model.encode_image(image_observation)
#     state /= state.norm(dim=-1, keepdim=True)

#     # g
#     goal_text = clip.tokenize(['win the game', 'navigate to the goal']).to(device)
#     target:torch.Tensor = model.encode_text(goal_text)
#     target /= target.norm(dim=-1, keepdim=True)
#     target = target.mean(dim=0, keepdim=True)
    
#     # b
#     baseline_text = clip.tokenize(['maze', 'game', 'navigation']).to(device)
#     baseline:torch.Tensor = model.encode_text(baseline_text)
#     baseline /= baseline.norm(dim=-1, keepdim=True)
#     baseline = baseline.mean(dim=0, keepdim=True)

# #
# # (2, 512)
# # torch.Size([2, 512])
# # (1, 512)
# # torch.Size([3, 512])
# # (1, 512)


#     # R
#     alpha = 0.5
#     direction = target - baseline
#     projection = direction.T @ direction / torch.norm(direction) ** 2
#     identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
#     projection = alpha * projection + (1 - alpha) * identity
#     rewards = 1 - (torch.norm((state - target) @ projection, dim=-1) ** 2) / 2

# print(rewards)
# print(rewards[0].item())