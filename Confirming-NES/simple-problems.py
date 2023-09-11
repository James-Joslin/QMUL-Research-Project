import time
import torch
import torch.optim as optim
import torch.nn as nn
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns
os.environ["RAY_DEDUP_LOGS"] = "0"

# NES file
import nes

# Ray config
import ray
NUM_RUNS = 20
NUM_STEPS = 2000
INPUT = 10
LAYER_SIZE = 8
OUTPUT = 2
PLOT_WIDTH = 3.25
PLOT_HEIGHT = 4.05
DPI = 225
RERUN = True

if RERUN:
    df_list = []

    # Define the Sphere function
    def sphere_func(x):
        return torch.sum(x ** 2)

    # Define the Rastrigin function
    def rastrigin(x):
        A=10 # hard-coded A parameter
        n = x.size(0)
        sum_term = torch.sum(x**2 - A*torch.cos(2*math.pi*x))
        return A*n + sum_term

    ### Adam
    # Define a simple feed-forward network with 2 outputs
    class Net(nn.Module):
        def __init__(self, input, layer_size, output):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input, layer_size)  # Input layer
            self.fc2 = nn.Linear(layer_size, output)   # Output layer

        def forward(self, x):
            x = torch.relu(self.fc1(x)) 
            x = self.fc2(x)
            return x

    for run in range(NUM_RUNS):
        # Initialize the network and optimizer - Sphere Function - Adam
        print("===SPHERE FUNCTION USING ADAM OPTIMISER===")
        adam_net = Net(
            input = INPUT,
            layer_size = LAYER_SIZE,
            output = OUTPUT
        )
        optimizer = optim.Adam(lr = 0.001, params=adam_net.parameters())
        sphere_vals = []

        for i in range(NUM_STEPS):  
            optimizer.zero_grad()

            # Random input to the network
            input = torch.randn(INPUT)

            # Output of the network is the solution x to the optimization problem
            x = adam_net(input)

            # Evaluate the Sphere function with x
            sphere_val = sphere_func(x)
            sphere_vals.append(sphere_val.item())

            # Backpropagation
            loss = sphere_val
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}: adam_sphere_val = {sphere_val}")

        # Initialize the network and optimizer - Rastrigin function - Adam
        print("===RASTRIGIN FUNCTION USING ADAM OPTIMISER===")
        adam_net = Net(
            input = INPUT,
            layer_size = LAYER_SIZE,
            output = OUTPUT
        )
        optimizer = optim.Adam(lr = 0.001, params= adam_net.parameters())
        rast_vals = []

        for i in range(NUM_STEPS):
            optimizer.zero_grad()

            # Random input to the network
            input = torch.randn(INPUT)

            # Output of the network is the solution x to the optimization problem
            x = adam_net(input)

            # Evaluate the Rastrigin function with x
            rastrigin_val = rastrigin(x)
            rast_vals.append(rastrigin_val.item())

            # Backpropagation
            loss = rastrigin_val
            loss.backward()
            optimizer.step()

            # if i % 100 == 0:
            #     print(f"Iteration {i}: adam_rastrigin_val = {rastrigin_val.item()}")
                
        # Initialize the network and optimizer - Sphere Function - SGD
        print("===SPHERE FUNCTION USING SGD OPTIMISER===")
        sgd_net = Net(
            input = INPUT,
            layer_size = LAYER_SIZE,
            output = OUTPUT
        )
        optimizer = optim.SGD(lr = 0.001, params=sgd_net.parameters())
        sgd_sphere_vals = []

        for i in range(NUM_STEPS): 
            optimizer.zero_grad()

            # Random input to the network
            input = torch.randn(INPUT)

            # Output of the network is the solution x to the optimization problem
            x = sgd_net(input)

            # Evaluate the Sphere function with x
            sphere_val = sphere_func(x)
            sgd_sphere_vals.append(sphere_val.item())

            # Backpropagation
            loss = sphere_val
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}: sgd_sphere_val = {sphere_val}")

        # Initialize the network and optimizer - Rastrigin function - Adam
        print("===RASTRIGIN FUNCTION USING SGD OPTIMISER===")
        sgd_net = Net(
            input = INPUT,
            layer_size = LAYER_SIZE,
            output = OUTPUT
        )
        optimizer = optim.SGD(lr = 0.001, params= sgd_net.parameters())
        sgd_rast_vals = []

        for i in range(NUM_STEPS):
            optimizer.zero_grad()

            # Random input to the network
            input = torch.randn(INPUT)

            # Output of the network is the solution x to the optimization problem
            x = sgd_net(input)

            # Evaluate the Rastrigin function with x
            rastrigin_val = rastrigin(x)
            sgd_rast_vals.append(rastrigin_val.item())

            # Backpropagation
            loss = rastrigin_val
            loss.backward()
            optimizer.step()

            # if i % 100 == 0:
            #     print(f"Iteration {i}: sgd_rastrigin_val = {rastrigin_val.item()}")

        ### NES
        # Initialize the network and optimizer - sphere function - NES
        ray.init(num_cpus=4)
        print("===SPHERE FUNCTION USING NES OPTIMISER===")
        nes_net = nes.SimpleNet(
            input = INPUT,
            layer_size = LAYER_SIZE,
            output = OUTPUT
        )
        # summary(nes_net, input_size=(INPUT,))
        nes_optimser = nes.nes(
            nes_net,
            num_es_workers = 4, 
            generation_es = 4, 
            population_size_es = 4, 
            noise_std_dev_es = 0.1, 
            learning_rate_es = 0.005,
            std_decay = 1,
            lr_decay = 1,
            to_optimise = sphere_func
        )
        nes_sphere_vals = []
        time.sleep(0.2)
        for i in range(NUM_STEPS):
            input = torch.randn(INPUT)
            updated_parameters = nes_optimser.optimise(input)
            
            nes_net.set_nn_parameters(updated_parameters)
            x = nes_net(input)
            sphere_val = sphere_func(x)
            nes_sphere_vals.append(sphere_val.item())
            
            if i % 100 == 0:
                print(f"Iteration {i}: nes_sphere_val = {sphere_val}")
        ray.shutdown()

        #Initialize the network and optimizer - Rastrigin function - NES
        ray.init(num_cpus=8)  
        # print("===RASTRIGIN FUNCTION USING NES OPTIMISER===")
        nes_net = nes.SimpleNet(
            input = INPUT,
            layer_size = LAYER_SIZE,
            output = OUTPUT
        )
        # summary(nes_net, input_size=(INPUT,))
        nes_optimser = nes.nes(
            nes_net,
            num_es_workers = 4, 
            generation_es = 3, 
            population_size_es = 8, 
            noise_std_dev_es = 0.0155, 
            learning_rate_es = 0.0005,
            std_decay = 1,
            lr_decay = 1,
            to_optimise = rastrigin
        )
        nes_rast_vals = []
        time.sleep(0.2)
        for i in range(NUM_STEPS):
            input = torch.randn(INPUT)
            updated_parameters = nes_optimser.optimise(input)
            
            nes_net.set_nn_parameters(updated_parameters)
            x = nes_net(input)
            nes_rast_val = rastrigin(x)
            nes_rast_vals.append(nes_rast_val.item())
            
            if i % 100 == 0:
                print(f"Iteration {i}: nes_rast_val = {nes_rast_val}")
        ray.shutdown()

        # Convert lists to pandas DataFrame
        data = pd.DataFrame({
            'Iteration': list(range(NUM_STEPS)) * 2 + list(range(NUM_STEPS)) * 2 + list(range(NUM_STEPS)) * 2,
            'Loss': sphere_vals + rast_vals + sgd_sphere_vals + sgd_rast_vals + nes_sphere_vals + nes_rast_vals,
            'Function': ['Sphere'] * NUM_STEPS + ['Rastrigin'] * NUM_STEPS + ['Sphere'] * NUM_STEPS + ['Rastrigin'] *  NUM_STEPS + ['Sphere'] * NUM_STEPS + ['Rastrigin'] * NUM_STEPS,
            'Optimiser': ['Adam'] * (NUM_STEPS*2) + ['SGD'] * (NUM_STEPS*2) + ['NES'] * (NUM_STEPS*2),
            'Run': [run] * NUM_STEPS * 6 
        })
        df_list.append(data)

    all_data = pd.concat(df_list)
    all_data.to_csv("./OptimiserData.csv", header=True, index=False)

else:
    all_data = pd.read_csv("./OptimiserData.csv", header=0)

unique_optimisers = all_data['Optimiser'].unique()
colors = ['red', 'green', 'blue']
# Create a dictionary to map optimisers to colors
color_mapping = {optimiser: colors[i % len(colors)] for i, optimiser in enumerate(unique_optimisers)}

fig, axes = plt.subplots(2, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=DPI, sharex=True)

corner_labels = ['A', 'B']
for ax, label, (function, sub_data) in zip(axes, corner_labels, all_data.groupby('Function')):
    for _, optimiser_sub_data in sub_data.groupby('Optimiser'):
        sns.lineplot(
            x=optimiser_sub_data['Iteration'].to_numpy(), 
            y=optimiser_sub_data['Loss'].to_numpy(), 
            ax=ax, 
            color=color_mapping[optimiser_sub_data['Optimiser'].iloc[0]], 
            label=optimiser_sub_data['Optimiser'].iloc[0],
            data=optimiser_sub_data,
            ci = 95,
            alpha=0.35,
            n_boot=1000
        )
    ax.set(xlabel='Epoch', ylabel='Loss') # title=f'{function} - {LAYER_SIZE} Nodes', 
    ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
    ax.legend(fontsize = 8)
# Adjust the padding between and around the subplots
# plt.subplots_adjust(hspace=0.01, wspace=0.01)
plt.tight_layout()
plt.show()
fig.savefig(f'./SphereRastrigin_{LAYER_SIZE}NodeLayer.png', dpi=DPI, pad_inches = 0)
