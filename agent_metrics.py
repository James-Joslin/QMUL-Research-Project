import pandas as pd
from glob import glob
import numpy as np
from scipy import integrate

files = glob("./parsed-data/*")
metrics = pd.DataFrame(columns=[
    "method", "domain", "mean", "median", "std", "kurtosis", "skew", "max", "min", "integral"
])
for i, dataset in enumerate(files):
    df = pd.read_csv(dataset, header=0)
    experiment = [df['method'].unique()[0], df['domain'].unique()[0]]
    
    individual_metrics = pd.DataFrame(
        columns=["mean", "median", "std", "kurtosis", "skew", "max", "min", "integral"]
    )
    for j, (item, group) in enumerate(df.groupby('id')):
        mean = group['total_reward'].mean()
        median = group['total_reward'].median()
        std = group['total_reward'].std()
        kurtosis = group['total_reward'].kurtosis()
        skew = group["total_reward"].skew()
        maxi = group["total_reward"].max()
        mini = group['total_reward'].min()
        
        values = group["total_reward"].to_numpy()
        integral = integrate.simpson(values)
        
        individual_metrics.loc[j] = [mean, median, std, kurtosis, skew, maxi, mini, integral]
    print(experiment)
    print(individual_metrics)
    mean_metrics = individual_metrics.mean().to_numpy().tolist()
    experiment.extend(mean_metrics)
    # print(experiment)
    metrics.loc[i] = experiment    
for item, group in metrics.groupby('domain'):
    print(group.sort_values('method'))
    print("\n")