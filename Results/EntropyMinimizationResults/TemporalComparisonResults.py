import pandas
import seaborn as sns
import matplotlib.pyplot as plt

# Mix results #

metric = 'Entropy'
df_drl = pandas.read_csv('./DRL_temporal_noisy.csv')
df_drl['Algorithm'] = 'DRL'
df_lm = pandas.read_csv('./TemporalLawnMowerResults.csv')
df_lm['Algorithm'] = 'Lawn Mower'
df_rn = pandas.read_csv('./TemporalFullRandom.csv')
df_rn['Algorithm'] = 'Random'
df_rc = pandas.read_csv('./TemporalRandomCoverage.csv')
df_rc['Algorithm'] = 'Coverage'
df_gr = pandas.read_csv('./Temporal_std_greedy.csv')
df_gr['Algorithm'] = 'Greedy $\sigma$'

all_data = pandas.concat([df_drl,df_lm,df_rn,df_rc,df_gr], ignore_index=True)
all_data['Distance'] = all_data['Distance']*45/200
all_data['A_vector'] = all_data['A_vector']*(0.225**2)

sns.set_theme()
sns.set(rc={'figure.figsize':(6,2)})
g = sns.lineplot(data=all_data, x='Distance', y=metric, hue='Algorithm', style='Algorithm')
plt.legend(fontsize=10)

g.set(ylabel=r'Information $I(X)$')
g.set(xlabel='Distance $(km)$')
g.set(xlim=(0, 45*5/2))


final_metrics = all_data.loc[all_data['Distance'] >= 45]

for algorithm_name in final_metrics['Algorithm'].unique():

	algorithm_metrics = final_metrics.loc[(final_metrics['Algorithm'] == algorithm_name)]
	print(f"###### ALGORITHM {algorithm_name} #######")

	for metric_name in ['Reward', 'Entropy', 'A_vector', 'DetectionRate', 'MSE_GP', 'MSE_SVR']:

		print(f'Value of {metric_name}: {algorithm_metrics[metric_name].mean()} +- {algorithm_metrics[metric_name].std()}')


plt.show()
