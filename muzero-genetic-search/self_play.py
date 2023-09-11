import time
import numpy
import ray
import torch
import models
import random
from graphviz import Digraph

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

                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0, # temperature = 0
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

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())
        # step = 0
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
                    # Runs the genetic search method to build and aggreagate optimised trajectories into a tree structure
                    root = genetic_search(self.config).run_search(
                        model = self.model,
                        observation= stacked_observations,
                        legal_actions= self.game.legal_actions(),
                        to_play= self.game.to_play()
                        
                    )
                    '''
                    Below is the graph build function, it builds a graphical representation of the search tree
                    This method is commented out and was only used for debug
                    There seems to be compatibility issues with ray and the graphing function, a single call of the function
                    will break the ray process
                    It serves it's purpose though by giving a representation of the tree after the first execution of the genetic search
                    Uncomment if you would like to see an example of the tree
                    Tree is rendered as a text file, copy and paste this file's contents into graphviz online or set up pathing for Graphviz
                    '''
                    # graph = self.build_graph(root, None)
                    # graph.render(filename='./debug_tree/output', format='png')
                    
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
            root = genetic_search(self.config).run_search(
                model = self.model,
                observation= stacked_observations,
                legal_actions= self.game.legal_actions(),
                to_play= self.game.to_play()
            )
            # print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. Genetic MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
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
        A form of boltzmann exploration that uses node visit counts to build a distribution
        Select action according to the visit count distribution and the temperature.
        The temperature is controlled with the visit_softmax_temperature function in the config.
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
        # print(action)
        return action

    # Used for debugging purposes - call to this function in play game is commented out
    def build_graph(self, node, graph=None):
        if graph is None:
            graph = Digraph("DecisionTree", format="png")

        label = (
            # f"Action: {node.action_taken}\n"
            f"Reward: {node.reward}\n"
            # f"Value: {node.value}\n"
            f"Value Sum: {node.value_sum}\n"
            # f"State: {node.hidden_state}\n"
            f"Visit Count: {node.visit_count}\n"
            f"To Play: {node.to_play}"
        )
        graph.node(str(id(node)), label)

        for edge_label, child in node.children.items():
            graph.edge(str(id(node)), str(id(child)), label=str(edge_label))
            self.build_graph(child, graph)

        return graph

## Sequenctial/Synchronous Evolutionary Search 
class genetic_search:
    def __init__(self, config) -> None:
        self.config = config

    def run_search(
        self,
        model, 
        observation,
        legal_actions,
        to_play
    ):        
        # create an initial population of random hypothetical action sequences to optimise
        population = self.create_population(
            legal_actions,
            self.config.num_actions,
            self.config.population_size
        )
        
        # prep observation as tensor with required shape for model
        observation = (
            torch.tensor(observation)
            .float()
            .unsqueeze(0)
            .to(next(model.parameters()).device)
        )
        # return predicted value, reward, policy logits and the initial hidden state
        (root_predicted_value,
            reward,
            policy_logits,
            intial_hidden_state,
        ) = model.initial_inference(observation) # if you see models model, initial inference utilises the representation function to create s^0
        # The prediction is used to create the initial value but this is not used in the search and is largely inconsequential
        # Reward is a log of a torch.zeros array (this was implemented by the original repository owners - and was kept so the data structure of the tree
        # remains consistent to the original MCTS tree)
        root_predicted_value = models.support_to_scalar(
            root_predicted_value, self.config.support_size
        ).item()
        reward = models.support_to_scalar(reward, self.config.support_size).item()

        for _ in range(self.config.num_generations): # For loop for g number of generations - set in config file
            # Evaluate population fitness and return a list of fitness scores
            fitness_scores = self.fitness_function(
                population=population,
                intial_state=intial_hidden_state,
                mu_model=model
            )
            
            # Pair fitness scores with their respective individual 
            fitness_scores = [(individual, score) for individual, score in zip(population, fitness_scores)]
            
            # Selection pressure on original population - survival of the fittest
            selected_population = self.select_survivors(
                fitness_scores,
                tournament_size=self.config.tournament_size, # ignored if chosen method is roulette
                method="tournament" # can also be "roulette" but roulette was less reliable as negative value functions would need to be offset
            )
            
            # Update population with next generation - calls crossover and mutation functions
            population = self.create_offspring(
                legal_actions,
                selected_population,
                self.config.mutation_rate,
                self.config.gs_dirichlet_alpha,
                True
            )
            
        # Once generations are complete, build a list of trajectories from the optimised policiles 
        # where each resultant state, reward etc... from an action is stored in a dictionary
        final_trajectories = self.create_trajectories(
                population,
                model,
                intial_hidden_state,
                to_play
            )
        # Build decision tree from the trajectories, where the nodes contain the relevant information
        root = self.build_tree(
            trajectories = final_trajectories, 
            discount = self.config.discount,
            initial_reward = reward,
            intial_hidden_state=intial_hidden_state)
        return root
    
    def create_population(
        self, 
        legal_actions, 
        num_actions,
        population_size
        ) -> list:
        # List comprehension to create N random action arrays of length K
        population = [random.choices(legal_actions, k=num_actions) for _ in range(population_size)]
        return population
    
    def fitness_function(self, population, intial_state, mu_model):
        # Initialise fitness array of zeroes
        fitness_values = numpy.zeros(len(population))
        for i in range(len(population)): # Iterate over the whole population
            rewards = [] # Reset returned rewards
            current_state = intial_state # reset state back to the initial encoded state
            policy = population[i] # iterate through random policies
            for j in range(len(policy)):
                # Pass the hidden state through the model to get an action
                action = policy[j]
                value, reward, policy_logits, hidden_state = mu_model.recurrent_inference( # recurrent inference method uses prediction and dynamics functions
                    current_state,
                    torch.tensor([[action]]).to(current_state.device)
                )
                value, reward = map(lambda x: models.support_to_scalar(x, self.config.support_size).item(),
                                    (value, reward))
                rewards.append(reward)
                current_state = hidden_state
            # Calculate the fitness value by summing rewards and the value of the final resultant state
            fitness_values[i] = sum(rewards) + value
        return fitness_values
    
    def select_survivors(
        self,
        fitness_scores : list, # will be a list of (policy, fitness) tuples
        tournament_size = 10, # default is 10, but actual tournament size provided from config file
        method = "tournament"
        ) -> list:
        selected_population = []
        match method:
            case "tournament": # chosen method
                for i in range(len(fitness_scores)):
                    tournament = random.sample(fitness_scores, tournament_size) # randomly select tuples based on the tournament size
                    best_individual = max(tournament, key=lambda x: x[1]) # select the tuple of subset with highest fitness 
                    selected_population.append(best_individual[0]) # append element one of tuple (the policy) to survivors
            
            case "roulette": # poor performance
                total_fitness = sum([pair[0] for pair in fitness_scores])
                selection_probs = [pair[0] / total_fitness for pair in fitness_scores]
                # Offset selection probs
                selection_probs = (selection_probs - numpy.min(selection_probs)) / (numpy.max(selection_probs) - numpy.min(selection_probs))
                selection_probs = selection_probs * (1 - 0.1) + 0.1
                for i in range(len(fitness_scores)):
                    chosen_index = numpy.random.choice(range(len(fitness_scores)), p=selection_probs)
                    selected_population.append(fitness_scores[chosen_index][1])
            
            case _:
                raise ValueError('Invalid method for selection')
        
        return selected_population
           
    def create_offspring( # creates next generation to be used to update initial population
        self,
        legal_actions,
        selected_population : list,
        mutation_rate : float,
        dirichlet_alpha : float,
        use_dirichlet_noise = True
        ) -> list:
        
        new_population = [] # initialise population
        
        for i in range(len(selected_population)):
            parent1, parent2 = random.sample(selected_population, 2) # select two random survivors to act as parents
            child = self.uniform_crossover(parent1, parent2) # cross parents with uniform crossover
            child = self.mutate(child, legal_actions, mutation_rate, dirichlet_alpha, use_dirichlet_noise) # mutate child
            new_population.append(child) # append offspring policy to new population
        return new_population

    def uniform_crossover(self, parent1 : list, parent2 : list) -> list:
        # Create a mask (boolean array) with the same length as parent
        parent1, parent2 = numpy.array(parent1), numpy.array(parent2)
        mask = numpy.random.randint(0, 2, len(parent1), dtype=bool)  

        # Initialize the child
        child = numpy.zeros_like(parent1)

        # If mask[i] is True, the i-th gene comes from parent1, otherwise from parent2.
        child[mask] = parent1[mask]
        child[~mask] = parent2[~mask]

        return child.tolist()
  
    def cycles_crossover(self, parent1, parent2) -> list: # unfortunatly this method failed as it couldn't be guarenteed that all actions would be in both parent policies  
        # Initialize the child as a list of None values
        child = [None]*len(parent1)

        # Choose a random starting index
        start = random.randint(0, len(parent1)-1)

        while None in child:
            # Create the cycle starting from the start index
            cycle = []
            i = start
            while i not in cycle:
                cycle.append(i)
                i = parent2.index(parent1[i])

            # For every index in the cycle, take the parent1 value if the index is even; parent2 otherwise
            for j in cycle:
                if cycle.index(j) % 2 == 0:
                    child[j] = parent1[j]
                else:
                    child[j] = parent2[j]

            # Choose a new start index from the indices that have not been included in a cycle yet
            if None in child:
                start = child.index(None)

        return child

    def mutate(
        self, 
        actions, 
        legal_actions,
        mutation_rate,
        dirichlet_alpha=0.25,
        use_dirichlet_noise = True
        ) -> list:
        if use_dirichlet_noise:
            # mutation rate defined in confif
            num_mutations = int(len(actions) * mutation_rate)
            # randomly indices to mutate at up to the number of mutations
            indices_to_mutate = random.sample(range(len(actions)), num_mutations)
            
            # Create probabilty distribution with dirichlet noise, alpha defined in config
            mutation_probs = numpy.random.dirichlet([dirichlet_alpha] * len(legal_actions))
            
            new_actions = actions.copy()
            for i, idx in enumerate(indices_to_mutate):
                # select new action based on probability distribution from legal actions
                new_action = numpy.random.choice(legal_actions, p=mutation_probs)
                # substitute action in offspring policy for new action
                new_actions[idx] = new_action
            return new_actions
            
        else:
            for i in range(len(actions)):
                if random.random() < mutation_rate:
                    actions[i] = random.choice(legal_actions)
            return actions
        
    def create_trajectories(
        self,
        population,
        model,
        intial_hidden_state,
        to_play
    ):
        # create list to hold performance metrics of population
        trajectories = []
        for policy in population: # iterate over all policies
            
            policy_trajectory = []            
            current_state = intial_hidden_state # start at initial hidden state
            
            
            for action in policy: # iterate through actions of current policy
                
                value, reward, policy_logits, hidden_state = model.recurrent_inference( # again use dynamics and prediction functios
                    current_state,
                    torch.tensor([[action]]).to(current_state.device)
                )
                value, reward = map(lambda x: models.support_to_scalar(x, self.config.support_size).item(),
                                    (value, reward))
                
                policy_trajectory.append({ # build dictionary containing the action taken, value, reward, resultant state etc...
                    "action_taken": action,
                    "to_play":to_play,
                    "value": value,
                    "reward": reward,
                    "resultant_state": hidden_state
                })
                current_state = hidden_state # update current state to progress along trajectory
                
            trajectories.append(policy_trajectory)
    
        return trajectories

    def build_tree(
        self, 
        trajectories, 
        discount,
        initial_reward,
        intial_hidden_state
        ):
        
        root = EvoNode(0) # instantiate new node
        root.hidden_state = intial_hidden_state # populate teh root node with the initial hidden state, and the initial rewards
        root.reward = initial_reward # as pre-stated the intiial reward if a log torch.zeros tensor

        for trajectory in trajectories: # iterate throught the trajectories
            current_EvoNode = root # start at root and work down the steps of the current trajectory

            for i, step in enumerate(trajectory):
                action_taken = step['action_taken'] # get step data
                state = step['resultant_state']
                reward = step['reward']
                value = step['value']
                to_play = step['to_play']
                
                # if the action at this current step hasn't been taken before a new state node will need to be created
                # a nodes "children" is a dictionary of actions paired with the resultant node (state and metric info)
                # as of python 3.7 dictionaries are sorted data structures
                if action_taken not in current_EvoNode.children: 
                    child = EvoNode( # we create the child node and populate it with the important metrics
                        state=state,
                        reward=reward,
                        to_play=to_play,
                        parent=current_EvoNode,
                        depth = int(i+1))

                    # then it is added to the tree
                    # this entails adding the action to the current nodes children dictionary as a key, and the child node as the paired value
                    # see EvoNode class
                    current_EvoNode.add_child(child, action_taken) 

                # We then select the subsequent node of the action taken to be the current node, if the action wasn't in the parent's child action dictionary
                # then this will be the child we just made
                # If it was in the child action dictionary then the previous if statement is bypassed, no new child needs to be made and we just select the 
                # next state in the dictionary
                current_EvoNode = current_EvoNode.children[action_taken]
                
                # this process slowly builds a tree and ensures that new branches and sub-branches are only produced when a new action taken in the current
                # state arises 

                # Check if we're on the last step of the trajectory, if we are backpropogate
                if i == len(trajectory) - 1:
                    # Backpropagate from the leaf node
                    # increases visitation count, and updates value sums for the game history
                    # within the game history class the sum is averaged by the visitation count
                    current_EvoNode.backpropagate(value, discount)

        return root

class EvoNode: # Node class used by evolutionary search decision tree
    def __init__(self, state=None, reward=None, to_play=None, parent=None, depth = 0):
        self.visit_count = 0
        self.to_play = to_play
        self.value_sum = 0
        self.children = {}
        self.hidden_state = state
        self.reward = reward
        self.parent = parent
        self.depth = depth

    def add_child(self, child, action):
        child.parent = self
        self.children[action] = child

    def mean_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def backpropagate(self, value, discount):
        '''
        Backpropogation method starts at the node from which the methods was called
        Updates value sum and visit count - the same as the MCTS method
        Moves up trajectory by getting the parent of the current node and updating the current node to be the parent
        - Different from MCTS method, as the MCTS methods moves up the search path list of nodes
        '''
        current_node = self
        while current_node is not None:
            current_node.value_sum += value
            current_node.visit_count += 1

            if current_node.depth >= 0:
                value = current_node.reward + discount * value
                
            current_node = current_node.parent

## Game history class 
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

    def store_search_statistics(self, root, action_space):
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

# Below is a parallelised approach that was considered but not properly tested or implemented
# It was built to learn about and practice with Ray and Parallel processing before working on the NES approach
# I later improved the aggregation of subsets to remove the need for sorting
@ray.remote
class fitness_worker:
    def __init__(self, config) -> None:
        self.config = config
    
    def fitness_function(
        self,
        intial_state,
        mu_model,
        population_subset,
        indices_set
        ):
        
        fitness_values = numpy.zeros(len(population_subset))
        for i in range(len(population_subset)):
            rewards = []
            current_state = intial_state
            policy = population_subset[i]
            for j in range(len(policy)):
                # Pass the hidden state through the model to get an action
                action = policy[j]
                value, reward, policy_logits, hidden_state = mu_model.recurrent_inference(
                    current_state,
                    torch.tensor([[action]]).to(current_state.device)
                )
                value, reward = map(lambda x: models.support_to_scalar(x, self.config.support_size).item(),
                                    (value, reward))
                rewards.append(reward)
                current_state = hidden_state
            # Calculate the fitness value based on the chosen metric
            fitness_values[i] = sum(rewards) + value

        return fitness_values, indices_set
            
class parrallel_genetic_search:
    def __init__(self, config) -> None:
        self.config = config        
        self.num_eval_workers = config.num_gs_workers
        self.gs_workers = [fitness_worker.remote(config) for _ in range(self.num_eval_workers)]

    def run_search(
        self,
        model, 
        observation,
        legal_actions,
        to_play
    ):  
        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(self.config.action_space)
        ), "Legal actions should be a subset of the action space."
        initial_hidden_state, initial_reward = self.prepare_observation(model, observation)

        population = self.create_population(
            legal_actions,
            self.config.num_actions,
            self.config.population_size
        )
        
        for _ in range(self.config.num_generations):
            population_indices = list(range(len(population)))
            population_batches = numpy.array_split(population, self.num_eval_workers)
            index_batches = numpy.array_split(population_indices, self.num_eval_workers)

            fitness_scores_and_indices = ray.get(
                [gs_worker.fitness_function.remote(
                    intial_state = initial_hidden_state,
                    mu_model = model,
                    population_subset = batch,
                    indices_set = batch_indices) for gs_worker, batch, batch_indices in zip(self.gs_workers, population_batches, index_batches)])
            
            fitness_values, indices = zip(*fitness_scores_and_indices)
            fitness_values = numpy.concatenate(fitness_values)
            indices = numpy.concatenate(indices)
            
            # Sort the rewards based on the returned indices - makes sure that if the order that the workers return is not the same as they were called, we can reorder back to the original order
            sorted_order = numpy.argsort(indices)
            fitness_values = fitness_values[sorted_order]
            
            ranked_population = [(individual, score) for individual, score in zip(fitness_values, population)]
            
            selected_population = self.select_survivors(
                ranked_population,
                tournament_size=6,
                method="tournament"
            )
            
            population = self.create_offspring(
                legal_actions,
                selected_population,
                self.config.mutation_rate,
                self.config.gs_dirichlet_alpha,
                True
            )
            
        evolved_policy_trajectories = self.create_trajectories(
            population,
            model,
            initial_hidden_state,
            to_play
        )
        
        root = self.build_tree(evolved_policy_trajectories, self.config.discount)
        # graph = self.build_graph(root)
        return root #, graph
    
    def prepare_observation(self, mu_model, observation):
        # prep observation as tensor with required shape for model
        observation = (
            torch.tensor(observation)
            .float()
            .unsqueeze(0)
            .to(next(mu_model.parameters()).device)
        )
        # return predicted value, reward, policy logits and the hidden state - hidden state is the next encoded state after inference
        (root_predicted_value,
            reward,
            policy_logits,
            intial_hidden_state,
        ) = mu_model.initial_inference(observation)
        root_predicted_value = models.support_to_scalar(
            root_predicted_value, self.config.support_size
        ).item()
        reward = models.support_to_scalar(reward, self.config.support_size).item()
        
        return intial_hidden_state, reward
    
    def create_population(
        self, 
        legal_actions, 
        num_actions,
        population_size
        ) -> numpy.ndarray:
        
        population = numpy.array([random.choices(legal_actions, k=num_actions) for _ in range(population_size)])
        return population
        
    def select_survivors(
        self,
        ranked_population : list,
        tournament_size = 10,
        method = "tournament"
        ) -> numpy.ndarray:
        selected_population = []
        match method:
            case "tournament":
                for i in range(len(ranked_population)):
                    tournament = random.sample(ranked_population, tournament_size)
                    best_individual = max(tournament, key=lambda x: x[0])
                    selected_population.append(best_individual[1])
            
            case _:
                raise ValueError('Invalid method for selection')
        # print("Selected Population")
        return numpy.array(selected_population)
    
    def create_offspring(
        self,
        legal_actions,
        selected_population : list,
        mutation_rate : float,
        dirichlet_alpha : float,
        use_dirichlet_noise = True
        ) -> numpy.ndarray:
        
        new_population = []
        unique_parents = self.get_unique_parents(selected_population)
        
        for i in range(len(selected_population) - len(unique_parents)):
            parent_indices = numpy.random.choice(len(selected_population), 2, replace=False)
            parent1, parent2 = selected_population[parent_indices[0]], selected_population[parent_indices[1]]
            
            child = self.uniform_crossover(parent1, parent2)
            child = self.mutate(child, legal_actions, mutation_rate, dirichlet_alpha, use_dirichlet_noise)
            new_population.append(child)
        
        new_population.extend(unique_parents)
        # print("Created new Population")
        return numpy.array(new_population)

    def get_unique_parents(self, action_arrays):
        unique = []
        seen = set()
        for arr in action_arrays:
            key = hash(tuple(arr))
            if key not in seen:
                unique.append(arr)
                seen.add(key)
        return unique
    
    def cycles_crossover(self, parent1, parent2) -> list:
        # Initialize the child as a list of None values
        child = [None]*len(parent1)

        # Choose a random starting index
        start = random.randint(0, len(parent1)-1)

        while None in child:
            # Create the cycle starting from the start index
            cycle = []
            i = start
            while i not in cycle:
                cycle.append(i)
                i = numpy.where(parent2 == parent1[i])[0][0] 

            # For every index in the cycle, take the parent1 value if the index is even; parent2 otherwise
            for j in cycle:
                if cycle.index(j) % 2 == 0:
                    child[j] = parent1[j]
                else:
                    child[j] = parent2[j]

            # Choose a new start index from the indices that have not been included in a cycle yet
            if None in child:
                start = child.index(None)

        return child

    def uniform_crossover(self, parent1, parent2) -> list:
        # Create a mask (boolean array) with the same length as parent
        mask = numpy.random.randint(0, 2, len(parent1), dtype=bool)  

        # Initialize the child
        child = numpy.zeros_like(parent1)

        # If mask[i] is True, the i-th gene comes from parent1, otherwise from parent2.
        child[mask] = parent1[mask]
        child[~mask] = parent2[~mask]

        return child.tolist()

    def two_point_crossover(self, parent1, parent2) -> list:
        # Determine two random crossover points
        size = len(parent1)
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two crossover points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Create child by taking genes from first parent between the two points, 
        # and genes from second parent for the remaining genes
        child = numpy.hstack((parent1[:cxpoint1], parent2[cxpoint1:cxpoint2], parent1[cxpoint2:]))

        return child.tolist()

    def mutate(
        self, 
        actions, 
        legal_actions,
        mutation_rate,
        dirichlet_alpha=0.15,
        use_dirichlet_noise = True
        ) -> list:
        """
        Mutates an action sequence
        
        Args:
            actions (list): The action sequence to mutate.
            mutation_rate (float): The probability of a mutation occurring.
            legal_actions: pool of action that can be chosen if element of action sequence mutates
            dirichlet_alpha (float): The concentration parameter of the Dirichlet distribution.
            
        Returns:
            list: The mutated action sequence.
        """
        if use_dirichlet_noise:
            # add Dirichlet noise to the probabilities of selecting each action
            num_mutations = int(len(actions) * mutation_rate)
            indices_to_mutate = random.sample(range(len(actions)), num_mutations)
            mutation_probs = numpy.random.dirichlet([dirichlet_alpha] * len(legal_actions))
            
            new_actions = actions.copy()
            for i, idx in enumerate(indices_to_mutate):
                new_action = numpy.random.choice(legal_actions, p=mutation_probs)
                new_actions[idx] = new_action
            return new_actions
            
        else:
            for i in range(len(actions)):
                if random.random() < mutation_rate:
                    actions[i] = random.choice(legal_actions)
            return actions

    def create_trajectories(
        self,
        population,
        model,
        intial_hidden_state,
        to_play
    ):
        # create list to hold performance metrics of population
        trajectories = []
        for policy in population:
            # print(policy)
            policy_trajectory = []            
            current_state = intial_hidden_state
            
            for action in policy:
                # print(action)
                value, reward, policy_logits, hidden_state = model.recurrent_inference(
                    current_state,
                    torch.tensor([[action]]).to(current_state.device)
                )
                value, reward = map(lambda x: models.support_to_scalar(x, self.config.support_size).item(),
                                    (value, reward))
                
                policy_trajectory.append({
                    "visit_count":0,
                    "current_state": current_state,
                    "action_taken": action,
                    "to_play":to_play,
                    "value": value,
                    "reward": reward,
                    "policy_logits": policy_logits,
                    "resultant_state": hidden_state
                })
                current_state = hidden_state
                
            trajectories.append(policy_trajectory)
    
        return trajectories

    def build_tree(self, trajectories, discount):
        root = ParEvoNode(0)

        for trajectory in trajectories:
            current_EvoNode = root

            for i, step in enumerate(trajectory):
                action_taken = step['action_taken']
                state = step['current_state']
                reward = step['reward']
                value = step['value']
                to_play = step['to_play']

                if action_taken not in current_EvoNode.children:
                    child = ParEvoNode(
                        action=action_taken,
                        state=state,
                        reward=reward,
                        value=value,
                        to_play=to_play,
                        parent=current_EvoNode,
                        depth = int(i+1))

                    current_EvoNode.add_child(child, action_taken)

                current_EvoNode = current_EvoNode.children[action_taken]

                # Check if we're on the last step of the trajectory
                if i == len(trajectory) - 1:
                    # Backpropagate from the leaf node
                    current_EvoNode.backpropagate(value, discount)

        return root

class ParEvoNode:
    def __init__(self, action=None, state=None, reward=None, value=None, to_play=None, parent=None, depth = 0):
        self.action = action
        self.visit_count = 0
        self.to_play = -1
        self.value_sum = 0
        self.children = {}
        self.hidden_state = state
        self.reward = 0
        self.parent = parent
        self.individual_value = 0
        self.depth = depth

    def add_child(self, child, action):
        child.parent = self
        self.children[action] = child

    def mean_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def backpropagate(self, value, discount):
        current_node = self
        while current_node is not None:
            current_node.value_sum += value
            current_node.visit_count += 1

            if current_node.depth >= 0:
                value = current_node.reward + (discount ** current_node.depth) * value
                
            current_node = current_node.parent
