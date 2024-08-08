from VLMRM import CLIPReward

from PIL import Image
import time

print(type(CLIPReward))


# model, preprocess = clip.load('ViT-B/16', device=device)

vlmrm = CLIPReward(clip_model_name='ViT-B/16',
                   is_state_image=True, is_target_image=False, is_baseline_image=False,
                   device='cuda:0'
                   )

vlmrm.set(target_raw=['win the game', 'navigate to the goal'],
          baseline_raw=['maze', 'game', 'navigation'],
          alpha=0.5
          )
for i in range(10):
    s = time.time()
    rewards = vlmrm.get_rewards(image_observations=[Image.open(f"Usage/images/maze1.jpeg"),
                                                    # Image.open(f"Usage/images/maze2.jpeg"),
                                                    ])
    
    print(rewards)
    print(time.time() - s)