from VLMRM import CLIPReward
from VLMRM.utils import root_dir

from PIL import Image

def test_installation():
    print()
    vlmrm = CLIPReward(clip_model_name='ViT-B/16', 
                       is_state_image=True, is_target_image=False, is_baseline_image=False, 
                       device='auto')
    
    target_raw = ['win the game', 'navigate to the goal']
    baseline_raw = ['maze', 'game', 'navigation']
    alpha = 0.5
    vlmrm.set(target_raw=target_raw,
            baseline_raw=baseline_raw,
            alpha=alpha
            )
    print("target_raw =", target_raw)
    print("baseline_raw =", baseline_raw)
    print("alpha =", alpha)
    
    images = [root_dir + "/image/maze1.jpeg",
              root_dir + "/image/maze2.jpeg"]
    image_observations = [Image.open(image) for image in images]
    rewards = vlmrm.get_rewards(observations=image_observations)
    print()
    print("images = ")
    for image in images:
        print(image)
    print()
    print("rewards =", rewards)
    print()
    
if __name__ == "__main__":
    test_installation()