# Testing Alternative Decision Methods to Replace MuZero's Monte Carlo Tree Search

This project explores various decision-making algorithms as potential replacements for the Monte Carlo Tree Search (MCTS) utilized in MuZero (Schrittwieser et al., 2020).

## Installation

1. Ensure you have **Python 3.10.7** installed.
2. It's recommended to use a virtual environment. You can set it up using:
    ```bash
    python3 -m venv venv_name
    source venv_name/bin/activate
    ```
3. Once you're in your virtual environment, clone the repository and navigate to its directory.
4. Install the required packages using:
    ```bash
    pip install -r requirements.txt
    ```

> **Note**: This project was tested within an **Ubuntu 23.04 (Lunar Lobster)** OS environment.

## Base Repository

This project is an extension of the [MuZero-General](https://github.com/werner-duvaud/muzero-general) repository by Werner Duvard.

## About the Modifications

In this research, alternative search methods were explored to see if they could replace the original MCTS in MuZero. The initial experiments were done in separate branches, and after thorough testing, the methods were added to the main repository. You can find these methods in their respective directories, each building upon the `MuZero-MCTS-Search` framework.

### MuZero with Beam Search

1. **Overview**: 
   - This method constructs a heuristic derived trajectory based on MuZero's components.
   
2. **Modified Files and Directories**:
   - Changes within: `./muzero-beam-search/`
   - Adjusted files: 
     - `games/gridworld.py` and `games/lunarlander.py`: Additional config hyperparameters have been added.
     - `self-play.py`: MCTS has been removed, and the Beam Search method has been implemented in its place.
     - `diagnose_model.py`: Adjusted to diagnose MuZero models and plot trajectories that have been trained with Beam Search.

### MuZero with Genetic Search

1. **Overview**: 
   - This method creates optimized trajectories via a genetic search and then aggregates the trajectories into a decision tree.
   
2. **Modified Files and Directories**:
   - Changes within: `./muzero-genetic-search/`
   - Adjusted files: 
     - `controller_trainer.py`: A new file that contains the NES optimizer. It's responsible for optimizing the controller network through the generation of state trajectories via perturbed parameters.
     - `self_play.py`: The search tree has been removed. Instead, it loads the controller network and requests updated parameters for each self-play game it plays from the shared storage.
     - `games/gridworld.py` and `games/lunarlander.py`: Additional config hyperparameters have been added.
     - `muzero.py`: Modified to call the controller trainer worker. Adjustments have been made to the checkpoint dictionary to include the controller network's parameters as a state dictionary. The pretrained model functionality has been updated to load the controller network from the `.checkpoint` file, allowing for inference without further training.
     - `shared_storage.py`: Modified to correctly store the controller parameters for saving to the `.checkpoint` file. Additionally, it distributes the controller parameters to self-play agents when they access the shared storage, updating the controller parameters within their self-play session.

### MuZero with NES (Salimans et al., 2017) Optimised Controller

1. **Overview**: 
   - This method obfuscates any decision tree. Instead, a new worker optimizes a controller neural network. The self-play agents then use this controller to make decisions and fill MuZero's replay buffer.
   
2. **Modified Files and Directories**:
   - Changes within: `./muzero-nes/`
   - Adjusted files: 
     - `controller_trainer.py`: A new file that contains the NES optimizer. It's responsible for optimizing the controller network through the generation of state trajectories via perturbed parameters.
     - `self_play.py`: The search tree has been removed. Instead, it loads the controller network and requests updated parameters for each self-play game it plays from the shared storage.
     - `games/gridworld.py` and `games/lunarlander.py`: Additional config hyperparameters have been added.
     - `muzero.py`: Modified to call the controller trainer worker. Adjustments have been made to the checkpoint dictionary to include the controller network's parameters as a state dictionary. The pretrained model functionality has been updated to load the controller network from the `.checkpoint` file, allowing for inference without further training.
     - `shared_storage.py`: Modified to correctly store the controller parameters for saving to the `.checkpoint` file. Additionally, it distributes the controller parameters to self-play agents when they access the shared storage, updating the controller parameters within their self-play session.


## Note on Sample Models

All search methods contain sample models within their `results` directories. These models can be loaded to render self-play games.

## Other Files & Utilities

- **Inference Utilities**:
  - `/testing-beamsearch/main.py`: Calls the MuZero sample model trained with Beam Search and the Beam Search methods from `muzero-beam-search`. It can be used to run a ray tuning event or to run and render the LunarLander-V2 environment. The following arguments can be parsed:
    ```bash
    --num_env_repeats: Number of times an environment is run within a tuning sample before taking an average or the number of times to test inference.
    --num_tuning_samples: Number of tuning samples to take from the search space.
    --num_cores: Number of parallel sampling runs - one per core.
    --record: Whether test results should be recorded.
    ```
  - `/testing-mcts/main.py`: Calls the MuZero sample model trained with MCTS and only runs inference in the LunarLander-V2 environment. The following arguments can be parsed:
    ```bash
    --num_env_repeats: Number of times an environment is run to test inference.
    --record: Whether the results should be saved or not - boolean value of 0 or 1 for no or yes, respectively.
    ```
  - `agent_metrics.py`: Used to determine tabulated performance metrics.
  - `graphing_inference`: Boxplot graphed using `.npz` files after running inference events in MCTS and Beam Search code (LunarLander-V2).
  - `plot_training_results.py`: Used to smooth training data and plot.

## Data

The project contains an empty directory for data (`parsed-data`). The necessary data can be found at [this location](https://github.qmul.ac.uk/ec22045/DissertationCode/tree/main/parsed-data). If you wish to look at the analytics, the data will need to be downloaded and unzipped into the `parsed-data` directory.

## References

1. Salimans, T., Ho, J., Chen, X., Sidor, S., Sutskever, I., 2017. Evolution Strategies as a Scalable Alternative to Reinforcement Learning. [Link](https://doi.org/10.48550/arXiv.1703.03864)
2. Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., Lillicrap, T., Silver, D., 2020. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. Nature 588, 604–609. [Link](https://doi.org/10.1038/s41586-020-03051-4)
3. Werner Duvaud, A.H., 2019. MuZero General: Open Reimplementation of MuZero. GitHub repository.
