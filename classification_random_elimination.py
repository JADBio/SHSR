# Load libraries
import numpy as np
import pandas as pd
from random import sample
from random import seed

# Select classification
analysis = 'classification'
# Data directory
data_directory = 'data/'
output_directory = 'results/'
# Load data
my_results = pd.read_csv(data_directory + 'ML_results_' + analysis + '.csv')
my_metadata = pd.read_csv(data_directory + 'metadata_' + analysis + '.csv')
my_datasets = my_metadata['dataset']
# # # Randomness and experiments parameters
# Random selection percentage
for random_selection_percentage in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    # Train-test split percentage
    train_percentage = 0.9
    # Define number of repetitions
    number_of_runs = 20
    # Define results table
    results_on_performance_update_vs_not = pd.DataFrame(columns=['perf_ratio', 'time_ratio'],
                                                        data=np.zeros((number_of_runs, 2)))
    # Aggregation columns
    aggregation_columns = ['dataset', 'time_fs', 'SES_maxK_2', 'SES_maxK_3', 'SES_alpha_0.01', 'SES_alpha_0.05',
                           'SES_alpha_0.10', 'LASSO_0', 'LASSO_0.25', 'LASSO_0.5', 'LASSO_1.0', 'LASSO_1.25',
                           'LASSO_1.5', 'LASSO_2.0']
    group_columns = ['dataset', 'SES_maxK_2', 'SES_maxK_3', 'SES_alpha_0.01', 'SES_alpha_0.05', 'SES_alpha_0.10',
                     'LASSO_0', 'LASSO_0.25', 'LASSO_0.5', 'LASSO_1.0', 'LASSO_1.25', 'LASSO_1.5', 'LASSO_2.0']
    # Run random configuration sampling
    for run_ in range(len(results_on_performance_update_vs_not)):
        # Split in train and test
        np.random.seed(run_)
        train_indices = np.sort(np.random.choice(range(len(my_datasets)), int(train_percentage * len(my_datasets)),
                                                 replace=False))
        train_datasets = np.array(my_datasets)[train_indices]
        if train_percentage == 1:
            test_datasets = train_datasets.copy()
        else:
            test_datasets = np.array(my_datasets)[np.in1d(my_datasets, train_datasets) == 0]
        train_metadata = my_metadata.loc[np.in1d(my_metadata['dataset'], train_datasets)]
        if train_percentage == 1:
            test_metadata = train_metadata.copy(deep=True)
        else:
            test_metadata = my_metadata.loc[np.in1d(my_metadata['dataset'], test_datasets)]
        train_results = my_results.copy().loc[np.in1d(my_results['dataset'], train_datasets)]
        if train_percentage == 1:
            test_results = train_results.copy(deep=True)
        else:
            test_results = my_results.copy().loc[np.in1d(my_results['dataset'], test_datasets)]
        # Calculate full set execution times
        execution_times_fs_test = test_results[aggregation_columns].groupby(group_columns).agg('max')
        execution_times_fs_sum_test = execution_times_fs_test.groupby(['dataset']).agg('sum')
        execution_times_ml_sum_test = test_results[['dataset', 'time_ml']].groupby(['dataset']).agg('sum')
        execution_times_sum_original_test = execution_times_fs_sum_test['time_fs'] +\
            execution_times_ml_sum_test['time_ml']
        execution_times_sum_original_test = execution_times_sum_original_test[test_datasets]
        # Calculate full set performance
        max_performance_original_test = test_results[['dataset', 'score']].groupby('dataset').agg('max')['score']
        max_performance_original_test = max_performance_original_test[test_datasets]
        # Random selection
        test_results_r = test_results.copy()
        test_results_r['select'] = False
        for d in range(len(test_datasets)):
            dataset = test_datasets[d]
            selection_vector = test_results_r.index[test_results_r['dataset'] == dataset]
            selection_n = int(random_selection_percentage * len(selection_vector))
            seed(1234 + 1000*run_ + d)
            selection_sample = sample(list(selection_vector), selection_n)
            test_results_r.at[selection_sample, 'select'] = True
        test_results_r = test_results_r.loc[test_results_r['select']]
        # Check start to end differences
        # Calculate execution times
        execution_times_fs = test_results_r[aggregation_columns].groupby(group_columns).agg('max')
        execution_times_fs_sum = execution_times_fs.groupby(['dataset']).agg('sum')
        execution_times_ml_sum = test_results_r[['dataset', 'time_ml']].groupby(['dataset']).agg('sum')
        execution_times_sum = execution_times_fs_sum['time_fs'] + execution_times_ml_sum['time_ml']
        execution_times_sum_r = execution_times_sum[test_datasets]
        # Calculate performance
        max_performance_r = test_results_r[['dataset', 'score']].groupby('dataset').agg('max')['score']
        # Save results
        results_on_performance_update_vs_not.at[run_, 'perf_ratio'] =\
            np.mean(max_performance_r / max_performance_original_test)
        results_on_performance_update_vs_not.at[run_, 'time_ratio'] =\
            sum(execution_times_sum_r) / sum(execution_times_sum_original_test)
        # Print result
        print('----- Random - Run', run_, '-----')
        print('Average performance ratio:', np.round(np.mean(max_performance_r / max_performance_original_test), 5))
        print('Ratio of time sum:', round(sum(execution_times_sum_r) / sum(execution_times_sum_original_test), 5))
        print('-----------------------------------------')
        print('-----------------------------------------')

    # Save results
    results_on_performance_update_vs_not.to_csv(output_directory + analysis + '_' +\
                                                str(random_selection_percentage) + '_selection.csv')
