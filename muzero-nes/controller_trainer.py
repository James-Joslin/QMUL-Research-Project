import math
import numpy
import ray
import torch
import models
import time

@ray.remote
class SelfPlay:    
    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()
        
        self.controller = models.SimpleNet( # Define controller architecture - used for making descisions that progress discrete time steps after an optimisation phase
            input_size=self.config.encoding_size,
            layer_size=self.config.layer_size_es,
            output_size=len(self.config.action_space)
        )
        self.controller.load_state_dict(initial_checkpoint["controller_weights"]) # initially determined in muzero.py
        self.nes_optimiser = nes(self.config, self.controller) # initialise the optimiser, providing config data, and the controller architecture to optimise
        # see the nes class for more details
        # print(self.controller.state_dict())
        
    def continuous_self_play(self, shared_storage):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))
            
            controller_parameters = self.play_game(
                self.config.visit_softmax_temperature_fn(
                    trained_steps=ray.get(
                        shared_storage.get_info.remote("training_step")
                    )
                ),
                self.config.temperature_threshold,
                False,
                "self",
                0,
            )                
            # Send the paramters to the shared storage worker so the other self play workers can utilise them with their instances of the controller architecture
            # and feed the MuZero replay buffer
            print(f'{controller_parameters} - From Controller Trainer')
            shared_storage.set_controller_weights.remote(controller_parameters)
            # Don't add game history to replay buffer
            if self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
        self.close_game()

    def play_game(self, temperature, temperature_threshold, render, opponent, muzero_player):
        """
        Play one game with actions dervied from controller that is getting optimised within this worker
        """
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )

                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play():
                    
                    parameters, initial_encoded_state = self.nes_optimiser.optimise( # call optimise function that uses muzero model to optimise the controller
                        mu_model=self.model, observation=stacked_observations) # optimistation process requires the production of state trajectories via the muzero components
                    # the optimiser also returns the initial encoded state (via the representation function) that the trajectories were built from
                    
                    self.controller.set_nn_parameters(parameters) # update the controller that is outside of the optimiser with the returned updated parameters
                    action_probs = self.controller(initial_encoded_state).flatten() # use the controller to return action probabilities
                    action = torch.argmax(action_probs).item() # the environment is progressed with just an argmax of the controller network

                else:
                    action = self.select_opponent_action(
                        opponent, stacked_observations
                    )

                observation, reward, done = self.game.step(action) # advance the environment one step so that a new optimisation parse can begin on the next hidden state

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                # # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation) # required to get the stacked observations
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        return self.controller.state_dict() # on the completion of a game, return the controllers current parameters as a state dict - not a vector
        # keeping it as a state dict makes it easier to assign parameters within other self-play workers that feed MuZero's buffer
        # game history is not returned to be sent to replay buffer - instead it will be reset during the next game

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            observation = (
                torch.tensor(stacked_observations)
                .float()
                .unsqueeze(0)
                .to(next(self.model.parameters()).device)
            )
            (
                root_predicted_value,
                reward,
                policy_logits,
                initial_hidden_state, # latent state of current observation - will be fed into the controller
            ) = self.model.initial_inference(observation)
            root_predicted_value, reward = map(lambda x: models.support_to_scalar(x, self.config.support_size).item(),
                                            (root_predicted_value, reward))
            
            # get action probabilities of the controller
            action_probs = self.controller(initial_hidden_state).flatten() # use the controller to return action probabilities
            print(f'Controller suggests action: {torch.argmax(action_probs)}')
            return self.game.human_to_action()
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

@ray.remote
class fitness_worker: # initialised by optimiser and held within optimiser
    def __init__(self, config, controller : models.SimpleNet) -> None:
        self.controller = controller # initilised with a copy of the controller - this copy will recieve parameter sets to test
        self.config = config # config data required for the support to scaler muzero function
    
    def evaluate_fitness(
        self,
        initial_state,
        mu_model,
        trajectory_length,
        parameter_batch,
        perturbation_batch
        ):
        with torch.no_grad():
            fitness_values = torch.zeros(len(parameter_batch)) # initialise the fitness vector of this worker
            for i in range(len(fitness_values)):
                rewards = []
                current_state = initial_state # each trajectory starts at the initial latent state
                self.controller.set_nn_parameters(parameter_batch[i])
                # Run the loop for num_iterations iterations
                for j in range(trajectory_length): # trajectory length defined in config as num_actions_es
                    # Pass the hidden state through the model to get an action
                    action = torch.argmax(self.controller(current_state).flatten()).item() # select with argmax, output has softmax function already built into forward pass
                    value, reward, policy_logits, hidden_state = mu_model.recurrent_inference( # utilise muzero components to determine, metrics and next encoded state
                        current_state,
                        torch.tensor([[action]]).to(current_state.device)
                    )
                    value, reward = map(lambda x: models.support_to_scalar(x, self.config.support_size).item(),
                                        (value, reward)) # support to scale normalisation
                    rewards.append(reward) # append to rewards
                    current_state = hidden_state
                # Calculate the fitness value based on the chosen metric
                fitness_values[i] = sum(rewards) + value # final fitness is cumulative rewards + final value

        return fitness_values, perturbation_batch # return fitness values, and corresponding perturbation batch of this worker - no need for offset parameters anymore
            
class nes:
    """
    Implements the Natural Evolution Strategies (NES) algorithm.
    
    Attributes:
        config: A configuration object containing parameters for the NES.
        num_eval_workers: Number of parallel fitness evaluation workers.
        nes_workers: A list of remote fitness workers.
        parameters: Current set of parameters that will be perturbed to create candidates.
    """
    def __init__(self, config, controller : models.SimpleNet) -> None:
        """
        Initializes the NES object with the given configuration and controller.
        
        Args:
            config: A configuration object.
            controller: An instance of the SimpleNet model.
        """
        self.config = config # store config data
        
        self.num_eval_workers = self.config.num_es_workers # initialise parallel fitness workers
        self.nes_workers = [fitness_worker.remote(config, controller) for _ in range(self.num_eval_workers)]

        self.parameters = controller.get_nn_parameters() # store initial parameters - will be updated on each completion of an optimisation cycle
        
    def optimise(
        self,
        mu_model, # Muzero model
        observation):
        """
        Optimises the parameters using the NES algorithm.
        
        Args:
            mu_model: The MuZero model used for inference.
            observation: Current observation of the environment.
        
        Returns:
            Updated parameters and the initial hidden state.
        """
        initial_hidden_state = self.prepare_observation(mu_model, observation) # encode initial hidden state to build trajectories from
        
        for _ in range(self.config.generation_es):
            candidate_solutions, perturbations = self.create_candidates( # perturb current stored set of parameters to create candidates for fitness evaluation
                self.config.noise_std_dev_es,
                self.config.population_size_es,
                self.parameters
            )
            solution_batches = numpy.array_split(candidate_solutions, self.num_eval_workers) # split the noise vectors and paramter sets into sub-batches
            perturbation_batches = numpy.array_split(perturbations, self.num_eval_workers)
            
            fitness_and_perturbations = ray.get( # distribute parameter sets and noise sub batches to the fitness worker, one per worker
                [nes_worker.evaluate_fitness.remote(
                    initial_hidden_state,
                    mu_model,
                    self.config.num_actions_es,
                    parameter_batch,
                    perturbation_batch) for nes_worker, parameter_batch, perturbation_batch in zip(self.nes_workers, solution_batches, perturbation_batches)])
            
            all_fitness_values = torch.cat([fp[0] for fp in fitness_and_perturbations]) # once sub batches for fitness values and perturbation noises have been returned aggregate them
            all_perturbations = torch.cat([fp[1] for fp in fitness_and_perturbations])
            
            '''
            normalise fitness scores - not used because some domains can return the same fitness for whole population
            meaning that all weights will be a multiplication of zero - salimans et al did use normalisation, but it resulted in 
            problems for this very reason - as noted in the disadvantages of their study
            '''
            # print(all_fitness_values)
            # normalised_fitness = (all_fitness_values - torch.mean(all_fitness_values))/ (torch.std(all_fitness_values) + 1e-10) epsillon value to 
            # print(normalised_fitness)
            
            # equation for updating parameters - see salimans et al. 2017 evolutionary strategies as a scalable alternative to tradional reinforcement learning 
            weighted_perturbations = all_fitness_values[:, None] * all_perturbations
            self.parameters = self.parameters + (self.config.learning_rate_es * (1 / (self.config.population_size_es * self.config.noise_std_dev_es)) * weighted_perturbations.sum(axis=0))

        return self.parameters, initial_hidden_state
         
    def prepare_observation(self, mu_model, observation):
        """
        Prepares the observation as a tensor and returns the initial hidden state.
        
        Args:
            mu_model: The MuZero model used for inference.
            observation: Current observation of the environment.
        
        Returns:
            The initial hidden state.
        """
        # prep observation as tensor with required shape for model
        observation = (
            torch.tensor(observation)
            .float()
            .unsqueeze(0)
            .to(next(mu_model.parameters()).device)
        )
        # return predicted value, reward, policy logits and the hidden state - hidden state is the next encoded state after inference
        (
            root_predicted_value,
            reward,
            policy_logits,
            intial_hidden_state,
        ) = mu_model.initial_inference(observation)
        root_predicted_value, reward = map(lambda x: models.support_to_scalar(x, self.config.support_size).item(),
                                        (root_predicted_value, reward))
        return intial_hidden_state
    
    def create_candidates(self, noise_std_dev, population_size, parameters):
        """
        Creates candidate solutions by perturbing the current parameters.
        
        Args:
            noise_std_dev: Standard deviation of the noise used for perturbations.
            population_size: Size of the population to be generated.
            parameters: Current set of parameters.
        
        Returns:
            A tuple containing the candidate weight vectors and the perturbations.
        """
        # Sample perturbations from the Gaussian distribution
        perturbations = torch.randn((population_size, parameters.shape[0])) * noise_std_dev
        # Create candidate solutions by adding the perturbations to the current parameters
        weight_vectors = parameters + perturbations
        return weight_vectors, perturbations
       
class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = numpy.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations
