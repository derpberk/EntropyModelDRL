import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_EPSILON = pd.read_csv('/Users/samuel/Desktop/runs/run-Temporal_PER_dueling_epsilongreedy-tag-train_accumulated_reward.csv')
df_EPSILON['MA'] = df_EPSILON.rolling(window=100)['Value'].mean()
df_EPSILON['MS'] = df_EPSILON.rolling(window=100)['Value'].std()
df_EPSILON['Algorithm'] = '$\epsilon$-greedy'

df_NOISY = pd.read_csv('/Users/samuel/Desktop/runs/run-Temporal_Noisy-tag-train_accumulated_reward.csv')
df_NOISY['MA'] = df_NOISY.rolling(window=100)['Value'].mean()
df_NOISY['MS'] = df_NOISY.rolling(window=100)['Value'].std()
df_NOISY['Algorithm'] = 'Noisy Q'

df_SAFE = pd.read_csv('/Users/samuel/Desktop/runs/run-Temporal_Safe_Noisy-tag-train_accumulated_reward.csv')
df_SAFE['MA'] = df_SAFE.rolling(window=100)['Value'].mean()
df_SAFE['MS'] = df_SAFE.rolling(window=100)['Value'].std()
df_SAFE['Algorithm'] = 'Censoring Q'

df = pd.concat([df_EPSILON, df_NOISY, df_SAFE], ignore_index=True)
sns.set_theme('poster')
sns.set(rc={'figure.figsize':(6,4)})
sns.color_palette("rocket", as_cmap=True)
g = sns.lineplot(data=df, x='Step', y='MA', hue='Algorithm',style='Algorithm', linewidth=2, palette=['tab:red','tab:blue','tab:green'])
plt.fill_between(df_EPSILON['Step'][::20], df_EPSILON['MA'][::20]-df_EPSILON['MS'][::20], df_EPSILON['MA'][::20]+df_EPSILON['MS'][::20], color='tab:red', alpha=0.2, linewidth=0)
plt.fill_between(df_NOISY['Step'][::20], df_NOISY['MA'][::20]-df_NOISY['MS'][::20], df_NOISY['MA'][::20]+df_NOISY['MS'][::20], color='tab:blue', alpha=0.2, linewidth=0)
plt.fill_between(df_SAFE['Step'][::20], df_SAFE['MA'][::20]-df_SAFE['MS'][::20], df_SAFE['MA'][::20]+df_SAFE['MS'][::20], color='tab:green', alpha=0.2, linewidth=0)

g.set(ylim=(0,30))
g.set(xlim=(1000,9900))
g.set(ylabel='Accumulated Reward')
g.set(xlabel='Episodes')

plt.show()

df_EPSILON = pd.read_csv('DRL_epsilongreedy.csv')
df_EPSILON = df_EPSILON.loc[df_EPSILON['Distance'] >= 200]
df_EPSILON['Algorithm'] = '$\epsilon$-greedy'

df_NOISY = pd.read_csv('DRL_noisy.csv')
df_NOISY = df_NOISY.loc[df_NOISY['Distance'] >= 200]
df_NOISY['Algorithm'] = 'Noisy Q'

df_SAFE = pd.read_csv('DRL_safe.csv')
df_SAFE = df_SAFE.loc[df_SAFE['Distance'] >= 200]
df_SAFE['Algorithm'] = 'Censoring Q'

df = pd.concat([df_EPSILON, df_NOISY, df_SAFE], ignore_index=True)

sns.boxplot(data=df, x='Algorithm', y='Entropy', showfliers=False, palette=['tab:red', 'tab:blue', 'tab:green'])
sns.swarmplot(data=df, x='Algorithm', y='Entropy', color="0.25", size=3)

sns.set(rc={'figure.figsize':(6,2)})

plt.show()