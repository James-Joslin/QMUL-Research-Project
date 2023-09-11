import math
import time
import numpy
import ray
import torch
import models

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
        
        self.controller = models.SimpleNet( # Define controller architecture
            input_size=self.config.encoding_size,
            layer_size=self.config.layer_size_es,
            output_size=len(self.config.action_space)
        )
        self.controller.load_state_dict(initial_checkpoint["controller_weights"])
        # print(self.controller.state_dict())
        
    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))
            
            # Load parameters for controller from shared storage, that have been produced by the controller trainer
            # game play by the self play agent is delayed within muzero.py, to allow controller trainer worker enough time
            # to save the first instance of the controller parameters
            self.controller.load_state_dict(ray.get(shared_storage.get_info.remote("controller_weights")))
            
            if not test_mode:
                game_history = self.play_game(
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
                # Save game played with controller model to replay buffer
                replay_buffer.save_game.remote(game_history, shared_storage)
                
            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def play_game(self, temperature, temperature_threshold, render, opponent, muzero_player):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
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
                    
                    observation = (
                        torch.tensor(observation)
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
                    
                    '''We use a node/tree structure to "trick" this implementation of MuZero into continuting, 
                    orginating from the initial_node, which holds the initial latent state
                    If you don't have anything to populate the game history with then naturally it will break
                    The game history expects a certain data structure - so we just create a "tree" with one level from the resultant action probailities of the 
                    controller - one has to question the validity of a method that requires certain tricks'''
                    
                    initial_node = EvoNode(
                        initial_hidden_state,
                        reward,
                        root_predicted_value
                    )
                    
                    # enumerate the probability vector from the controller, the indices from the enumeration can act as out actions
                    for action, prob in enumerate(action_probs): 
                        
                        value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                            initial_hidden_state,
                            torch.tensor([[action]]).to(initial_hidden_state.device)
                        ) # use the dynamics and prediction functions to get the rewards and values of the each action 
                        
                        value, reward = map(lambda x: models.support_to_scalar(x, self.config.support_size).item(),
                                            (value, reward))
                        
                        if action not in initial_node.children:
                            child = EvoNode( # store the resultant data in a node attached to the initial node
                                resultant_state=hidden_state,
                                reward=reward,
                                value=value,
                                to_play=self.game.to_play(),
                                parent=initial_node,
                                nes_prob_logit = prob) # also includes the probability of the action generated via the controller model - will be used by the game history to create the policy
                            initial_node.add_child(child, action)
                            child.backpropagate(value, self.config.discount)
      
                    action = torch.argmax(action_probs).item() # the environment is progressed with just an argmax of the controller network

                else:
                    action = self.select_opponent_action(
                        opponent, stacked_observations
                    )

                observation, reward, done = self.game.step(action)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(initial_node, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        return game_history

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
          
'''# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            (
                root_predicted_value,
                reward,
                policy_logits,
                hidden_state,
            ) = model.initial_inference(observation)
            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size
            ).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            root.expand(
                legal_actions,
                to_play,
                reward,
                policy_logits,
                hidden_state,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[action]]).to(parent.hidden_state.device),
            )
            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(search_path, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())

                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
'''
# Node to store controller output data in - required by game history class, which is used to send the games data to the replay buffer
class EvoNode:
    def __init__(self, resultant_state=None, reward=None, value=None, to_play=None, parent=None, nes_prob_logit = 0):
        self.state = resultant_state
        self.reward = reward
        self.value = value
        self.value_sum = 0
        self.visit_count = 0
        self.to_play = to_play
        self.children = {}
        self.nes_prob_logit = nes_prob_logit
        self.parent = parent

    def add_child(self, child, action):
        child.parent = self
        self.children[action] = child
        
    def mean_value(self):
        if self.visit_count == 0:
            return 0
        # print((self.value_sum, self.visit_count))
        return self.value_sum / self.visit_count

    def backpropagate(self, value, discount):
        current_node = self
        while current_node is not None:
            current_node.value_sum += value
            current_node.visit_count += 1

            if current_node.reward is not None:
                value = current_node.reward + discount * value
                
            current_node = current_node.parent
              
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

    def store_search_statistics(self, root : EvoNode, action_space):
        # Use prob logits from es_model to act as a policy
        if root is not None:
            self.child_visits.append( # old child visits list now holds probability logits
                [
                    root.children[a].nes_prob_logit
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )
            self.root_values.append(root.mean_value())
        else:
            self.root_values.append(None)

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
