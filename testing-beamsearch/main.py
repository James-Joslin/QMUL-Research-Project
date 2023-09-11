import gym
import numpy as np
import sys
import torch
import pathlib
import argparse
import os
import time

# Ray
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

parser = argparse.ArgumentParser(description="Script to test Beam Search MuZero")
parser.add_argument("--num_env_repeats", type=int, default=1,
                    help="Number of times an environment is run within a tuning sample before taking an average.")
parser.add_argument("--num_tuning_samples", type=int, default=1,
                    help="Number of tuning samples to take from the search space")
parser.add_argument("--num_cores", type=int, default=1,
                    help="Number of parallel sampling runs - one per core")
parser.add_argument("--record", type=int, default=1,
                    help="Whether test results should be recorded")
args = parser.parse_args()

global BEAM_MUZERO_DIR # Global parameters required by ray to tune
global TUNING
global RENDER
global NUM_ENVIRONMENT_REPEATS
NUM_ENVIRONMENT_REPEATS = args.num_env_repeats # The number of times an environment is run within a tuning sample before taking an average
NUM_TUNING_SAMPLES = args.num_tuning_samples # Number of random samples within search space
NUM_CPUS = args.num_cores # Parallelism factor
RECORD = args.record # Whether test results should be recorded

# FOR AHSA - Not used
MAX_T = 35 # Number of repeats before a tuning sample is terminated due to allow for a new sample - AHSA Only!
GRACE_PERIOD = 10 # Number of repeats before a tuning sample is terminated due to poor performance - AHSA Only!
REDUCTION_FACTOR = 2 # The number of trials to cut down on act each successive halving - AHSA Only!

def run_lunarlander(n_episodes, muzero, config):
    from self_play import BeamSearch
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
            if RENDER:
                env.render()
                # pass
            else:
                pass
            start = time.time()
            root = BeamSearch(config).search(
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
        
        if not TUNING:
            print(f'Episode {i_episode + 1} - Total Reward: {total_reward} - Current Average: {sum(all_rewards)/len(all_rewards)}')
            
    env.close()
    final_average = sum(all_rewards)/len(all_rewards) # seeking to maximise the final average
    return all_rewards, average_durations, final_average

def run_beam_muzero(config):
    # MuZero Imports - Within function as Ray Tuner requires the sys paths to be available to all workers
    sys.path.insert(1, BEAM_MUZERO_DIR)
    from games.lunarlander import MuZeroConfig
    import models
    
    NUM_EPISODES = NUM_ENVIRONMENT_REPEATS
    # MuZero Model Config
    muzero_config = MuZeroConfig()

    muzero_config.max_depth = 2
    muzero_config.beam_width = 2
    muzero_config.seed = 94
    
    if TUNING: # search space
        # muzero_config.seed = config['seed']
        muzero_config.root_dirichlet_alpha = config["root_dirichlet_alpha"]
        muzero_config.root_exploration_fraction = config["root_exploration_fraction"]
        muzero_config.reward_heuristic_discount = config["reward_heuristic_discount"]
        muzero_config.value_heuristic_discount = config["value_heuristic_discount"]
    else: # hard coded parameters
        muzero_config.root_dirichlet_alpha = 0.25
        muzero_config.root_exploration_fraction = 0.25
        muzero_config.reward_heuristic_discount = 1.0
        muzero_config.value_heuristic_discount = 1.0

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
    checkpoint_path = pathlib.Path(f'{BEAM_MUZERO_DIR}/results/lunarlander/LunarLander-BeamSearch-Sample/model.checkpoint')
    checkpoint = torch.load(checkpoint_path)
    muzero.set_weights(checkpoint["weights"])
    muzero.to(DEVICE)
    print(muzero.eval())

    all_rewards, average_durations, final_average = run_lunarlander(NUM_EPISODES, muzero, muzero_config)

    if TUNING:
        tune.report(final_average=final_average)
    else:
        return np.array(all_rewards), np.array(average_durations)

if __name__ == "__main__":
    BEAM_MUZERO_DIR = str(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'muzero-beam-search/')))
    if not os.path.isdir(BEAM_MUZERO_DIR):
        print("Directory to Beam-MuZero couldn't be found\nPlease make sure that this testing script's directory and the Beam MuZero directory are in the same parent directory")
    else:
        prompt = '''MuZero with Beam Search in LunarLander-V2\nIs this a tuning run?\n0: No\n1: Yes'''
        print(prompt)
        
        valid_inputs = [str(i) for i in range(0,2,1)]
        TUNING = input("Is this a tuning run [0/1]:")
        while TUNING not in valid_inputs:
            TUNING = input("In valid input, is this a tuning run:")
        TUNING = int(TUNING)
        RENDER = TUNING
        
        if not TUNING: # Standard run
            rewards, timings = run_beam_muzero(config = None)
            if RECORD:
                np.savez("./Inference-Beam-Search_Inference_Metrics", rewards, timings)
        
                npz = np.load("./Inference-Beam-Search_Inference_Metrics.npz")
                datasets = npz.files
                for dataset in datasets:
                    print(npz[dataset])    
            
        else: # Load tuning tools
            # Define the reporter with the desired columns
            reporter = CLIReporter(
                metric_columns=[
                    "status", "seed", "root_dirichlet_alpha",
                    "root_exploration_fraction", "reward_heuristic_discount",
                    "value_heuristic_discount", "final_average"])
            
            search_space = {
                # "seed": tune.randint(1,100),
                "root_dirichlet_alpha": tune.uniform(0.225,0.275),
                "root_exploration_fraction": tune.uniform(0.225,0.275),
                "reward_heuristic_discount": 1, # tune.uniform(0, 1),
                "value_heuristic_discount": 1 # tune.uniform(0, 1)
            }

            scheduler = ASHAScheduler( # Currently not used
                max_t=MAX_T, 
                grace_period=GRACE_PERIOD,  # Minimum number of iterations before a trial can be early stopped
                reduction_factor=REDUCTION_FACTOR  # Reduction factor for halving
            ) # used with random search - now moving to bayesian search

            ray.init(num_cpus=NUM_CPUS) # being parallelised tuning process
            analysis = tune.run(
                run_beam_muzero,
                # scheduler=scheduler,
                config=search_space,
                metric="final_average",
                mode="max",
                resources_per_trial={"cpu": 1},
                num_samples=NUM_TUNING_SAMPLES,  # how many random samples from the search space
                progress_reporter=reporter 
            )
            print("============Getting best config============")
            best_config = analysis.get_best_config(metric="final_average", mode="max")
            for key, value in best_config.items():
                print(f"{key}: {value}")
            print(f'Tested: {NUM_TUNING_SAMPLES} parameter samples\nWith: {NUM_ENVIRONMENT_REPEATS} repeat per sample')
            ray.shutdown()
            
