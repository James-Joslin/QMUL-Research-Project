import gym
import numpy as np
import os
import sys
import torch
import pathlib
import argparse
import time

parser = argparse.ArgumentParser(description="Script to test MCTS MuZero")
parser.add_argument("--num_env_repeats", type=int, default=1,
                    help="Number of times an environment is run within a tuning sample before taking an average.")
parser.add_argument("--record", type=int, default=1,
                    help="Whether the results should be saved or not - boolean value or 0 or 1 for no or yes respectively")
args = parser.parse_args()
NUM_ENVIRONMENT_REPEATS = args.num_env_repeats # The number of times an environment is run within a tuning sample before taking an average
RECORD = args.record

def run_lunarlander(n_episodes, muzero, config):
    all_rewards = []
    average_durations = []
    # Initialize the environment
    env = gym.make('LunarLander-v2')
    # Initialize numpy random seed
    np.random.seed(config.seed)
    for i_episode in range(n_episodes):    
        observation = env.reset()
        durations = []
        total_reward = 0
        done = False
        while not done:
            observation = np.array([[observation]])
            # Render the environment
            # env.render()
            start = time.time()
            root, _ = MCTS(config).run(
                model = muzero,
                observation=observation,
                legal_actions=list(range(4)),
                to_play = 0,
                add_exploration_noise=True,
                override_root_with=None
            )
            visit_counts = np.array([child.visit_count for child in root.children.values()], dtype="int32")
            action = [action for action in root.children.keys()][np.argmax(visit_counts)]
            observation, reward, done, info = env.step(action)
            end = time.time()
            
            selection_time = end - start
            total_reward += reward
            durations.append(selection_time)
            
        average_durations.append(sum(durations)/len(durations))   
        all_rewards.append(total_reward)
        
        print(f'Episode {i_episode + 1} - Total Reward: {total_reward} - Current Average: {sum(all_rewards)/len(all_rewards)}')
            
    env.close()
    return all_rewards, average_durations

def run_mcts_muzero(num_repeats, checkpoint_path) -> np.ndarray:
    # MuZero Model Config
    muzero_config = MuZeroConfig()
    
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        muzero_config.selfplay_on_gpu = True
        print("CUDA is available. Setting the device to GPU.")
    else:
        DEVICE = torch.device('cpu')
        muzero_config.selfplay_on_gpu = False
        print("CUDA is not available. Setting the device to CPU.")
        
    
    # Load MuZero with config and parameters
    torch.manual_seed(muzero_config.seed)
    muzero = models.MuZeroNetwork(muzero_config)
    checkpoint = torch.load(checkpoint_path)
    muzero.set_weights(checkpoint["weights"])
    muzero.to(DEVICE)
    print(muzero.eval())

    all_rewards, average_durations = run_lunarlander(num_repeats, muzero, muzero_config)
    return np.array(all_rewards), np.array(average_durations)

if __name__ == "__main__":
    
    MCTS_MUZERO_DIR = str(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'muzero-mcts-search/')))
    if not os.path.isdir(MCTS_MUZERO_DIR):
        print("Directory to orignal MuZero-General couldn't be found\nPlease make sure that this testing script's directory and the MuZero directory are in the same parent directory")
    else:
        sys.path.insert(1, MCTS_MUZERO_DIR)
        from games.lunarlander import MuZeroConfig
        import models
        from self_play import MCTS
        
        checkpoint_path = os.path.join(MCTS_MUZERO_DIR, "results/lunarlander/LunarLander-MCTS-Sample/model.checkpoint")
        rewards, timings = run_mcts_muzero(NUM_ENVIRONMENT_REPEATS, checkpoint_path)
        if RECORD:
            np.savez("./MCTS_Inference_Metrics", rewards, timings)
        
            npz = np.load("./MCTS_Inference_Metrics.npz")
            datasets = npz.files
            for dataset in datasets:
                print(npz[dataset])