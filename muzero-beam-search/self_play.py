import time
import numpy
import ray
import torch
import models
from graphviz import Digraph
from typing import Tuple, List

@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

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

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

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
                # game history saved to replay buffer
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

                # Save to the shared storage for plotting but not stored to replay buffer
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                # only single player games tested therefore not used be kept for code integrity:
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
                    root = BeamSearch(self.config).search(
                        model = self.model,
                        observation=stacked_observations,
                        legal_actions=self.game.legal_actions(),
                        to_play = self.game.to_play(),
                        add_exploration_noise=True,
                        override_root_with=None
                    )
                    '''
                    Like the genetic search method we can plot the decision tree of the beam search
                    The decision tree provides a singlular trajectroy of actions
                    MuZero general's diagnosis tools also serve the function of plotting a graph
                    Use the diagnosis tool to plot a tree of trained models
                    The graphs I was making here were to debug the first simulation event to confirm that the algorithm was working properly
                    '''
                    graph = self.build_graph(root)
                    graph.render("./debug_tree/output", format="png")
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )

                else:
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations
                    )

                observation, reward, done = self.game.step(action)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(root, self.config.action_space)

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
            root = BeamSearch(self.config).search(
                model = self.model,
                observation=stacked_observations,
                legal_actions=self.game.legal_actions(),
                to_play = self.game.to_play(),
                temperature = 0,
                add_exploration_noise=True,
                override_root_with=None
            )
            return self.game.human_to_action(), root
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

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action

    # Used for debugging purposes
    def build_graph(self, node, graph=None):
        if graph is None:
            graph = Digraph("DecisionTree", format="png")

        label = (
            # f"Action: {node.action}\n"
            f"Reward: {node.reward}\n"
            f"Value: {node.individual_value}\n"
            f"Value Sum: {node.value_sum}\n"
            # f"State: {node.state}\n"
            f"Visit Count: {node.visit_count}\n"
            f"Prior: {node.prior}\n"
            # f"To Play: {node.to_play}"
        )
        graph.node(str(id(node)), label)

        for edge_label, child in node.children.items():
            graph.edge(str(id(node)), str(id(child)), label=str(edge_label))
            self.build_graph(child, graph)

        return graph

# Beam Search
class BeamNode:
    def __init__(self, prior = None, parent = None, depth = 0):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.parent = parent
        self.individual_value = 0
        self.depth = depth

    def expanded(self):
        return len(self.children) > 0

    def mean_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state, value, depth):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state
        self.individual_value = value

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)} # match policy valyes ot actions
        for action, p in policy.items():
            self.children[action] = BeamNode(prior = p, parent = self, depth=depth) # assign policy (prior) value to the leaf node attached to the relevant action (edge)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys()) # get all actions
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions)) # create an array of noise using the dirichlet alpha, of length actions
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac # equation to perturb prior values based on noise

    def backpropagate(self, value, discount): # start at current node and back propogate up through parents
        current_node = self
        while current_node is not None:
            current_node.value_sum += value
            current_node.visit_count += 1

            if current_node.depth >= 0:
                value = current_node.reward + discount * value

            current_node = current_node.parent

class BeamSearch:
    def __init__(self, config):
        self.config = config
        self.max_depth = config.max_depth
        self.beam_width = config.beam_width

    def search(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None
    ) -> BeamNode:
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            # Initialise empty root node as the base of the search tree
            root = BeamNode(0)
            # prep observation as tensor with required shape for model
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            # return predicted value, reward, policy logits and the initial hidden state
            (
                root_predicted_value,
                reward,
                policy_logits,
                initial_hidden_state,
            ) = model.initial_inference(observation)# if you see models model, initial inference utilises the representation function to create s^0
            # The prediction is used to create the initial value but this is not used in the search and is largely inconsequential
            # Reward is a log of a torch.zeros array (this was implemented by the original repository owners - and was kept so the data structure of the tree
            # remains consistent to the original MCTS tree)
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
            # Expand the node, appending state, reward, value and creating the edges via a childre dictionary
            # Edges are attached to leaf nodes that are empty, except for the prior values of the actions that they are attached to
            root.expand(
                actions = legal_actions,
                to_play = to_play,
                reward = reward,
                policy_logits = policy_logits,
                hidden_state = initial_hidden_state,
                value = root_predicted_value,
                depth=0
            )
        # Add exploration noise to root action prior values (that are held in the leaf nodes)
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        current_node = root      
        current_depth = 0
        # Until we hit maximum depth and the current node isn't a leaf node do:
        while current_depth < self.max_depth and current_node.expanded():
            # Expand the top k children, return the best child and the "runners up" within the beam - see the relative function for more details
            best, others = self.expand_top_k_children(current_node, model, self.beam_width, current_depth+1)
            # Continue the trajectory with the best node
            current_node = best

            # Back propogate the "runner ups"
            if len(others) > 0:
                for node in others:
                    node.backpropagate(node.individual_value, self.config.discount)
            current_depth += 1
        
        # Once you've reached the maximum depth back propogate the "optimal" trajectory
        current_node.backpropagate(current_node.individual_value, self.config.discount)

        return root

    def expand_top_k_children(self, node : BeamNode, model, beam_width, depth) -> Tuple[BeamNode, List[BeamNode]]:
        parent_hidden_state = node.hidden_state # get the parent hidden state that the actions will be progressing from
        legal_actions = list(node.children.keys()) # get the legal actions
        policy_values = torch.softmax(
            torch.tensor([node.children[a].prior for a in legal_actions]), dim=0
        ).tolist() # get the prior values of the legal actions
        # to do this we return the value of the children dictionary, which are the nodes attached to the actions and then call .prior from the node for each action

        # Select the actions with the highest prior values amd cull the rest
        top_k_indices = sorted(range(len(policy_values)), key=lambda i: policy_values[i], reverse=True)[:beam_width]
        top_k_actions = [legal_actions[i] for i in top_k_indices]

        expanded_children = []
        for action in top_k_actions: # iterate over each action in top k actions
            child_node = node.children[action] # get the node attached to that action

            if not child_node.expanded():
                action_tensor = torch.tensor([[action]]).to(parent_hidden_state.device) # make sure action is on the same device as state

                value, reward, policy_logits, hidden_state = model.recurrent_inference(parent_hidden_state, action_tensor) # use dynamics and prediction functions to get rewards, values etc...
                value = models.support_to_scalar(value, self.config.support_size).item() # the scalar function normalises the metrics
                reward = models.support_to_scalar(reward, self.config.support_size).item()

                child_node.expand( # expand the child node to store rhe resultant rewards etc... and to add leaf nodes that will hold the prior values for the next actions
                    self.config.action_space,
                    node.to_play,
                    reward,
                    policy_logits,
                    hidden_state,
                    value,
                    depth
                )

                expanded_children.append(child_node) # append to a list 

        # Select the expanded child with the highest reward among the expanded children
        best_child = max(expanded_children, key=lambda child: (self.config.reward_heuristic_discount * child.reward) + (self.config.value_heuristic_discount * child.individual_value))
        other_children = [child for child in expanded_children if child != best_child]

        return best_child, other_children

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

    def store_search_statistics(self, root : BeamNode, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
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