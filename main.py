import gymnasium as gym
import ale_py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

gym.register_envs(ale_py)

def resize_observation(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    obs = obs.astype(np.float32) / 255.0
    return obs

def main():
    env = gym.make("ALE/Pong-v5")
    obs, info = env.reset()

    obs = resize_observation(obs)
    print(f"Min value: {obs.min()}, Max value: {obs.max()}")
    plt.imshow(obs, cmap='gray')
    plt.savefig('preprocessed_frame.png')
    plt.close()

    # Create a frame stack to hold the first 4 frames
    frame_stack = deque(maxlen=4)
    for _ in range(4):
        frame_stack.append(obs)
      
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")
    
    done = False
    step_count = 0
    episode_reward = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        obs = resize_observation(obs)
        frame_stack.append(obs)
        episode_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: Action={action}, Reward={reward}, Episode Total={episode_reward}")
    
    print(f"\nEpisode ended. Total reward: {episode_reward}")
    env.close()

    stacked = np.array(list(frame_stack))
    print(f"Stacked frames shape: {stacked.shape}")

if __name__ == "__main__":
    main()


