# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data_directory = 'data/'
results_directory = 'results/'
output_directory = 'plots/'
regression = pd.read_csv(data_directory + 'datasets_regression.csv')
classification = pd.read_csv(data_directory + 'datasets_classification.csv')
# Plot and save
# regression
plt.scatter(np.log10(regression.N_SAMPLES), np.log10(regression.N_FEATURES))
plt.grid()
plt.title('Regression datasets')
plt.xlabel('log samples')
plt.ylabel('log features ')
plt.savefig(output_directory + 'datasets_regression.pdf')
plt.clf()
# classification
plt.scatter(np.log10(classification.N_SAMPLES), np.log10(classification.N_FEATURES), c=classification.minority_f)
plt.grid()
plt.title('Classification datasets')
plt.xlabel('log samples')
plt.ylabel('log features ')
plt.savefig(output_directory + 'datasets_classification.pdf')
plt.clf()

# Results
# Configurations sub_sampling
# Set number of experiments run
N = 20
for task in ['classification', 'regression']:
    parameters = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = pd.DataFrame(columns=['parameter', 'performance_mean', 'performance_variance',
                                    'time_mean', 'time_variance'])
    results['parameter'] = parameters
    for i in range(len(results)):
        results_file = pd.read_csv(results_directory +\
                                   task + '_dt_0.999_thresh_' + str(parameters[i]) +\
                                   '_configurations_0.9_train.csv')
        results.at[i, 'performance_mean'] = results_file.mean()['perf_ratio_ho']
        results.at[i, 'performance_variance'] = results_file.var()['perf_ratio_ho']
        results.at[i, 'time_mean'] = results_file.mean()['time_ratio_ho']
        results.at[i, 'time_variance'] = results_file.var()['time_ratio_ho']
    plt.scatter(results['parameter'], results['performance_mean'], c='blue')
    plt.scatter(results['parameter'], results['time_mean'], c='orange')
    plt.legend(['Performance', 'Time'])
    plt.plot(results['parameter'], results['performance_mean'], c='blue', linestyle=':')
    plt.plot(results['parameter'], results['time_mean'], c='orange', linestyle=':')
    for i in range(len(results)):
        plt.plot([parameters[i], parameters[i]],
                 [results.loc[i, 'performance_mean'] - 1.96*np.sqrt(results.loc[i, 'performance_variance']/N),
                  results.loc[i, 'performance_mean'] + 1.96*np.sqrt(results.loc[i, 'performance_variance']/N)],
                 c='blue')
        plt.plot([parameters[i], parameters[i]],
                 [results.loc[i, 'time_mean'] - 1.96*np.sqrt(results.loc[i, 'time_variance']/N),
                  results.loc[i, 'time_mean'] + 1.96*np.sqrt(results.loc[i, 'time_variance']/N)],
                 c='orange')
    plt.grid()
    if task == 'classification':
        plt.title('Classification')
    else:
        plt.title('Regression')
    plt.xlabel('Percentage of Subsampling')
    plt.ylabel('Ratio')
    plt.savefig(output_directory + task + '_configuration_percentage.pdf')
    plt.clf()

# Performance threshold
# Set number of experiments run
N = 20
for task in ['classification', 'regression']:
    parameters = [0.95, 0.97, 0.99, 0.999, 0.9999]
    results = pd.DataFrame(columns=['parameter', 'performance_mean', 'performance_variance',
                                    'time_mean', 'time_variance'])
    results['parameter'] = parameters
    for i in range(len(results)):
        results_file = pd.read_csv(results_directory +
                                   task + '_dt_' + str(parameters[i]) + '_thresh_1.0_configurations_0.9_train.csv')
        results.at[i, 'performance_mean'] = results_file.mean()['perf_ratio_ho']
        results.at[i, 'performance_variance'] = results_file.var()['perf_ratio_ho']
        results.at[i, 'time_mean'] = results_file.mean()['time_ratio_ho']
        results.at[i, 'time_variance'] = results_file.var()['time_ratio_ho']
    plt.scatter(results['parameter'], results['performance_mean'])
    plt.scatter(results['parameter'], results['time_mean'])
    plt.legend(['Performance', 'Time'])
    plt.plot(results['parameter'], results['performance_mean'], c='blue', linestyle=':')
    plt.plot(results['parameter'], results['time_mean'], c='orange', linestyle=':')
    for i in range(len(results)):
        plt.plot([parameters[i], parameters[i]],
                 [results.loc[i, 'performance_mean'] - 1.96*np.sqrt(results.loc[i, 'performance_variance']/N),
                  results.loc[i, 'performance_mean'] + 1.96*np.sqrt(results.loc[i, 'performance_variance']/N)],
                 c='blue')
        plt.plot([parameters[i], parameters[i]],
                 [results.loc[i, 'time_mean'] - 1.96*np.sqrt(results.loc[i, 'time_variance']/N),
                  results.loc[i, 'time_mean'] + 1.96*np.sqrt(results.loc[i, 'time_variance']/N)],
                 c='orange')
    plt.grid()
    if task == 'classification':
        plt.title('Classification')
    else:
        plt.title('Regression')
    plt.xlabel('Threshold T')
    plt.ylabel('Ratio')
    plt.savefig(output_directory + task + '_algorithm_hp.pdf')
    plt.clf()

# Comparison with random elimination
# Set number of experiments run
N = 20
for task in ['classification', 'regression']:
    parameters = [0.95, 0.97, 0.99, 0.999, 0.9999]
    selection_percentages = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = pd.DataFrame(columns=['parameter', 'performance_mean', 'performance_variance',
                                    'time_mean', 'time_variance'])
    results['parameter'] = parameters
    results_random = pd.DataFrame(columns=['percentage', 'performance_mean', 'performance_variance',
                                           'time_mean', 'time_variance'])
    results_random['percentage'] = selection_percentages
    for i in range(len(results)):
        results_file = pd.read_csv(results_directory +
                                   task + '_dt_' + str(parameters[i]) + '_thresh_1.0_configurations_0.9_train.csv')
        # plt.scatter(results_file['time_ratio_ho_no'], results_file['perf_ratio_ho_no'], c='blue', s=10, alpha=0.3)
        results.at[i, 'performance_mean'] = results_file.mean()['perf_ratio_ho']
        results.at[i, 'performance_variance'] = results_file.var()['perf_ratio_ho']
        results.at[i, 'time_mean'] = results_file.mean()['time_ratio_ho']
        results.at[i, 'time_variance'] = results_file.var()['time_ratio_ho']
    for i in range(len(results_random)):
        results_file_random = pd.read_csv(results_directory +
                                          task + '_' + str(selection_percentages[i]) + '_selection.csv')
        # plt.scatter(results_file_random['time_ratio_no'], results_file_random['perf_ratio_no'],
        #             c='orange', s=10, alpha=0.3)
        results_random.at[i, 'performance_mean'] = results_file_random.mean()['perf_ratio']
        results_random.at[i, 'performance_variance'] = results_file_random.var()['perf_ratio']
        results_random.at[i, 'time_mean'] = results_file_random.mean()['time_ratio']
        results_random.at[i, 'time_variance'] = results_file_random.var()['time_ratio']

    plt.scatter(results['time_mean'], results['performance_mean'], c='blue')
    plt.scatter(results_random['time_mean'], results_random['performance_mean'], c='orange')
    plt.legend(['SHSR', 'Random'], loc='lower right')
    plt.plot(results['time_mean'], results['performance_mean'], c='blue', linestyle=':')
    plt.plot(results_random['time_mean'], results_random['performance_mean'], c='orange', linestyle=':')
    for i in range(len(results)):
        plt.plot([results.loc[i, 'time_mean'], results.loc[i, 'time_mean']],
                 [results.loc[i, 'performance_mean'] - 1.96*np.sqrt(results.loc[i, 'performance_variance']/N),
                  results.loc[i, 'performance_mean'] + 1.96*np.sqrt(results.loc[i, 'performance_variance']/N)],
                 c='blue', alpha=0.5)
        plt.plot([results.loc[i, 'time_mean'] - 1.96*np.sqrt(results.loc[i, 'time_variance']/N),
                  results.loc[i, 'time_mean'] + 1.96*np.sqrt(results.loc[i, 'time_variance']/N)],
                 [results.loc[i, 'performance_mean'], results.loc[i, 'performance_mean']],
                 c='blue', alpha=0.5)
    for i in range(len(results_random)):
        plt.plot([results_random.loc[i, 'time_mean'], results_random.loc[i, 'time_mean']],
                 [results_random.loc[i, 'performance_mean'] -
                  1.96*np.sqrt(results_random.loc[i, 'performance_variance']/N),
                  results_random.loc[i, 'performance_mean'] +
                  1.96*np.sqrt(results_random.loc[i, 'performance_variance']/N)],
                 c='orange', alpha=0.5)
        plt.plot([results_random.loc[i, 'time_mean'] - 1.96*np.sqrt(results_random.loc[i, 'time_variance']/N),
                  results_random.loc[i, 'time_mean'] + 1.96*np.sqrt(results_random.loc[i, 'time_variance']/N)],
                 [results_random.loc[i, 'performance_mean'], results_random.loc[i, 'performance_mean']],
                 c='orange', alpha=0.5)
    plt.grid()
    plt.xlabel('Time')
    if task == 'classification':
        plt.title('Classification')
    else:
        plt.title('Regression')
    plt.ylabel('Performance')
    plt.savefig(output_directory + task + '_algorithm_vs_random.pdf')
    plt.clf()
