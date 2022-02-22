import pandas
import seaborn as sns
import matplotlib.pyplot as plt

# Mix results #

metric = 'MSE_SVR'
df_drl = pandas.read_csv('./DRL_safe.csv')
df_drl['Algorithm'] = 'DRL'
df_lm = pandas.read_csv('./LawnMowerResults')
df_lm['Algorithm'] = 'Lawn Mower'
df_rn = pandas.read_csv('./StaticFullRandom.csv')
df_rn['Algorithm'] = 'Random'
df_rc = pandas.read_csv('./StaticRandomCoverage.csv')
df_rc['Algorithm'] = 'Coverage'
df_gr = pandas.read_csv('./std_greedy.csv')
df_gr['Algorithm'] = 'Greedy $\sigma$'

all_data = pandas.concat([df_drl,df_lm,df_rn,df_rc,df_gr], ignore_index=True)
all_data['Distance'] = all_data['Distance']*45/200

sns.set_theme()
sns.set(rc={'figure.figsize':(6,4)})
g = sns.lineplot(data=all_data, x='Distance', y=metric, hue='Algorithm', style='Algorithm')

g.set(ylabel=r'Mean Squared Error')
g.set(xlabel='Distance $(km)$')
g.set(xlim=(0, 45))
g.set(yscale='log')

final_metrics = all_data.loc[all_data['Distance'] >= 45]

for algorithm_name in final_metrics['Algorithm'].unique():

	algorithm_metrics = final_metrics.loc[(final_metrics['Algorithm'] == algorithm_name)]
	print(f"###### ALGORITHM {algorithm_name} #######")

	for metric_name in ['Reward', 'Entropy', 'A_vector', 'DetectionRate', 'MSE_GP', 'MSE_SVR']:

		print(f'Value of {metric_name}: {algorithm_metrics[metric_name].mean()} +- {algorithm_metrics[metric_name].std()}')


plt.show()
