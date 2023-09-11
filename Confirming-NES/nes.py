import torch
import torch.nn as nn
import numpy
import ray
import numpy

class SimpleNet(nn.Module):
    def __init__(self, input=0, layer_size=0, hidden_layers=0, output=0):
        super(SimpleNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Linear(input, layer_size)
        if hidden_layers > 0:
            self.hidden_layers_list = nn.ModuleList()
            for _ in range(hidden_layers):
                self.hidden_layers_list.append(nn.Linear(layer_size, layer_size))
        self.output_layer = nn.Linear(layer_size, output)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        if self.hidden_layers > 0:
            for hidden_layer in self.hidden_layers_list:
                x = torch.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x
    
    def get_nn_parameters(self):
        # Convert parameters to a flat tensor
        return torch.cat([param.view(-1) for param in self.parameters()])
    
    def set_nn_parameters(self, flat_parameters):
        idx = 0
        for param in self.parameters():
            size = numpy.prod(param.size())
            new_values = flat_parameters[idx:idx + size]
            param.data.copy_(new_values.view_as(param))
            idx += size
            
@ray.remote
class fitness_worker:
    def __init__(self, nes_model : SimpleNet, num : int, to_optimise) -> None:
        self.nes_model = nes_model
        self.fitness_function = to_optimise
        # print(f'WORKER {num} ONLINE')
    
    def evaluate_fitness(self, observation, parameter_set, perturbation_batch):
        fitness_values = torch.zeros(len(parameter_set))
        for i in range(len(fitness_values)):
            self.nes_model.set_nn_parameters(parameter_set[i])
            x = self.nes_model(observation)
            fitness_values[i] = -self.fitness_function(x) # negative value for fitness as we're trying to get to zero
        return fitness_values, perturbation_batch
            
class nes:
    def __init__(
        self, 
        nes_model : SimpleNet, 
        num_es_workers, 
        noise_std_dev_es, 
        population_size_es, 
        generation_es, 
        learning_rate_es,
        lr_decay,
        std_decay,
        to_optimise
        ) -> None:
        
        self.to_optimise = to_optimise
        self.num_eval_workers = num_es_workers
        self.es_workers = [fitness_worker.remote(nes_model, int(i), self.to_optimise) for i in range(self.num_eval_workers)]
        
        self.std = noise_std_dev_es
        self.population_size_es = population_size_es
        self.generation_es = generation_es
        self.learning_rate_es = learning_rate_es
        self.lr_decay = lr_decay
        self.std_decay = std_decay
        
        self.parameters = nes_model.get_nn_parameters()
        
        
    def optimise(self, observation) -> numpy.ndarray:
        for _ in range(self.generation_es):
            candidate_solutions, perturbations = self.create_candidates(
                self.std,
                self.population_size_es,
                self.parameters
            )
            solution_batches = numpy.array_split(candidate_solutions, self.num_eval_workers)
            perturbation_batches = numpy.array_split(perturbations, self.num_eval_workers)
            
            fitness_and_perturbations = ray.get(
                [es_worker.evaluate_fitness.remote(
                    observation,
                    solution_batch,
                    perturbation_batch) for es_worker, solution_batch, perturbation_batch in zip(self.es_workers, solution_batches, perturbation_batches)])
            
            all_fitness_values = torch.cat([fp[0] for fp in fitness_and_perturbations])
            all_perturbations = torch.cat([fp[1] for fp in fitness_and_perturbations])
            
            weighted_perturbations = all_fitness_values[:, None] * all_perturbations
            self.parameters = self.parameters + (self.learning_rate_es * (1 / (self.population_size_es * self.std)) * weighted_perturbations.sum(axis=0))

        self.learning_rate_es = self.learning_rate_es * self.lr_decay
        self.std = self.std * self.std_decay
        return self.parameters
        
    def create_candidates(self, noise_std_dev, population_size, parameters):
        # Sample perturbations from the Gaussian distribution
        perturbations = torch.randn((population_size, parameters.shape[0])) * noise_std_dev
        # Create candidate solutions by adding the perturbations to the current parameters
        weight_vectors = parameters + perturbations
        return weight_vectors, perturbations
