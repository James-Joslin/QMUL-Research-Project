import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import scipy.stats as stats

try:
    alchemyEngine = create_engine('postgresql+psycopg2://postgres:Lester134590@localhost:5432/postgres', pool_recycle=3600)
    dbConnection = alchemyEngine.connect()
    print("Connection established!")
except Exception as e:
    print("Unable to connect to the database.")
    print(e)

grid_tables = ["nes_gridworld", "genetic_gridworld", "mcts_gridworld", "beam_gridworld"]
lunar_tables = ["nes_lunarlander", "genetic_lunarlander", "mcts_lunarlander", "beam_lunarlander"]

domain_frames = []

def build_domain_frames(connection, tables:list):
    try:
        frames = []
        for table in tables:
        # Read the whole table and save result in a pandas dataframe
            df = pd.read_sql_table(table, connection)
            
            min_length = df.groupby('id').size().min()
            window_size = int(min_length * 0.01)
            trimmed_df = pd.DataFrame()
            for _, group in df.groupby('id'):
                trimmed_group = group.iloc[:min_length].copy().reset_index(drop=True)  # Create a copy to avoid warnings
                # Compute the rolling average
                trimmed_group.loc[:, 'rolling_avg'] = trimmed_group['total_reward'].rolling(window_size, 1).mean()
                print(trimmed_group.head(5))
                print(trimmed_group.shape)
                trimmed_df = pd.concat([trimmed_df, trimmed_group])
            
            frames.append(trimmed_df)
    except Exception as e:
        print("Error executing the query.")
        print(e)
    finally:
        df = pd.concat(frames)
        for i, domain in enumerate(df['domain'].unique()):
            subset = df[df['domain'] == domain]
            for method in df['method'].unique():
                method_subset = subset[subset['method'] == method]
                sums = []
                for item, group in method_subset.groupby('id'):
                    sums.append(group['total_reward'].sum())
                mean = sum(sums)/len(sums)
                std = np.array(sums).std()  
                print(f"Method being plotted: '{method}', Domain: {domain}, Average: {mean}, STD: {std}")
        return df

# window_size = 10 # replace with your actual window size
domain_frames.append(build_domain_frames(dbConnection, grid_tables))
domain_frames.append(build_domain_frames(dbConnection, lunar_tables))

dbConnection.close()

df = pd.concat(domain_frames)

plot_titles = ["GridWorld", "LunarLander"]
plot_legends = ['upper right', 'lower left']
fig, axs = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(5, 10))
plt.tight_layout()
for i, domain in enumerate(df['domain'].unique()):
    subset = df[df['domain'] == domain]
    for method in df['method'].unique():
        method_subset = subset[subset['method'] == method]
        sns.lineplot(ax=axs[i], x=method_subset.index, y='rolling_avg', data=method_subset, ci = 95, n_boot=1000, label=str(method), alpha = 0.65)

    axs[i].set_title(plot_titles[i])
    axs[i].legend(loc='upper left', prop={'size': 8})
    axs[i].set_xlabel('Game Episode')
    axs[i].set_ylabel('Rolling Average of Cumulative Reward')
    
    # Scientific notation
    format_x = lambda x, pos: f'{x/10000:.1f}'
    axs[i].xaxis.set_major_formatter(FuncFormatter(format_x))
    # axs[i].text(r'$\times10^4$', ha='lower right')
    
# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center')

plt.tight_layout()
plt.savefig("../Dissertation Paper - LATEX/Figures/ComparisonPlot.png", )
plt.show()


# g = sns.FacetGrid(df, col="id", col_wrap=2, height=2, aspect=1)
# # Map a regplot onto each facet
# g.map(sns.regplot, "step", "total_reward", order=5, scatter = True, scatter_kws={'s':1.25, 'alpha': 0.0025},
#       line_kws={'color':'red', 'linewidth' : 1})
# g.fig.suptitle('NES - LunarLander', y=0.02)
# g.set_titles(size='x-small')
# g.set_ylabels("Total Reward")
# g.set(ylim=(-200, 100))
# plt.tight_layout()
# plt.show()
# plt.ylim(-125, 125)  # Limits y-axis from -50 to 50