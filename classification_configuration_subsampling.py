# Load libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Data directory
data_directory = 'data/'
output_directory = 'results/'
# Load data
my_results = pd.read_csv(data_directory + 'ML_results_classification.csv')
my_metadata = pd.read_csv(data_directory + 'metadata_classification.csv')
my_datasets = my_metadata['dataset']
# # # Algorithm and experiments parameters
# Random configuration selection percentage
for configuration_percentage in [0.2, 0.4, 0.6, 0.8]:
    # Train-test split percentage
    train_percentage = 0.9
    # Set performance threshold
    thresh_perf = 0.999
    # Define number of repetitions
    number_of_runs = 20
    # Define results table
    results_on_performance_update_vs_not = pd.DataFrame(columns=['perf_ratio', 'time_ratio',
                                                                 'perf_ratio_ho', 'time_ratio_ho'],
                                                        data=np.zeros((number_of_runs, 4)))
    # Define Configuration subsets
    configurations_partition = np.concatenate([
        np.append(
            np.repeat(['SES_maxK_2', 'SES_maxK_3'], 3).reshape(-1, 1),
            np.tile(['SES_alpha_0.01', 'SES_alpha_0.05', 'SES_alpha_0.10'], 2).reshape(-1, 1),
            axis=1),
        np.append(
            np.repeat(['SVM_linear', 'SVM_rad', 'SVM_polynomial_2',
                       'SVM_polynomial_3', 'SVM_polynomial_4'], 1).reshape(-1, 1),
            np.tile(['SVM_linear', 'SVM_rad', 'SVM_polynomial_2',
                     'SVM_polynomial_3', 'SVM_polynomial_4'], 1).reshape(-1, 1),
            axis=1),
        np.append(
            np.repeat(['LASSO_0', 'LASSO_0.25', 'LASSO_0.5', 'LASSO_1.0', 'LASSO_1.25', 'LASSO_1.5', 'LASSO_2.0'],
                      1).reshape(-1, 1),
            np.tile(['LASSO_0', 'LASSO_0.25', 'LASSO_0.5', 'LASSO_1.0', 'LASSO_1.25', 'LASSO_1.5', 'LASSO_2.0'],
                    1).reshape(-1, 1), axis=1),
        np.array([['decisionTree', 'decisionTree']]),
        np.array([['randomForest', 'randomForest']]),
        np.array([['ridge', 'ridge']])])
    # Aggregation columns
    aggregation_columns = ['dataset', 'time_fs', 'SES_maxK_2', 'SES_maxK_3', 'SES_alpha_0.01', 'SES_alpha_0.05',
                           'SES_alpha_0.10', 'LASSO_0', 'LASSO_0.25', 'LASSO_0.5', 'LASSO_1.0', 'LASSO_1.25',
                           'LASSO_1.5', 'LASSO_2.0']
    group_columns = ['dataset', 'SES_maxK_2', 'SES_maxK_3', 'SES_alpha_0.01', 'SES_alpha_0.05', 'SES_alpha_0.10',
                     'LASSO_0', 'LASSO_0.25', 'LASSO_0.5', 'LASSO_1.0', 'LASSO_1.25', 'LASSO_1.5', 'LASSO_2.0']
    # Run algorithm
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
        # Randomly select configurations
        train_results['include'] = True
        for cd in range(len(train_datasets)):
            n_entries = np.sum(train_results.dataset == train_datasets[cd])
            np.random.seed((cd + 1) * (run_ + 1))
            selection = np.random.choice(n_entries, int(configuration_percentage*n_entries), replace=False)
            train_results.at[train_results.dataset == train_datasets[cd], 'include'] = np.in1d(range(n_entries),
                                                                                               sorted(selection))
        train_results = train_results.loc[train_results['include']]
        #############
        # Algorithm #
        #############
        # Define initial mapping
        mapping = list()
        for d in range(len(train_metadata)):
            mapping.append(configurations_partition)
        # Calculate full set execution times
        execution_times_fs = train_results[aggregation_columns].groupby(group_columns).agg('max')
        execution_times_fs_sum = execution_times_fs.groupby(['dataset']).agg('sum')
        execution_times_ml_sum = train_results[['dataset', 'time_ml']].groupby(['dataset']).agg('sum')
        execution_times_sum = execution_times_fs_sum['time_fs'] + execution_times_ml_sum['time_ml']
        execution_times_sum = execution_times_sum[train_datasets]
        execution_times_sum_original = execution_times_sum.copy(deep=True)
        #
        execution_times_fs_test = test_results[aggregation_columns].groupby(group_columns).agg('max')
        execution_times_fs_sum_test = execution_times_fs_test.groupby(['dataset']).agg('sum')
        execution_times_ml_sum_test = test_results[['dataset', 'time_ml']].groupby(['dataset']).agg('sum')
        execution_times_sum_original_test = execution_times_fs_sum_test['time_fs'] +\
            execution_times_ml_sum_test['time_ml']
        execution_times_sum_original_test = execution_times_sum_original_test[test_datasets]
        # Calculate full set performance
        max_performance = train_results[['dataset', 'score']].groupby('dataset').agg('max')['score']
        max_performance = max_performance[train_datasets]
        max_performance_original = max_performance.copy(deep=True)
        #
        max_performance_original_test = test_results[['dataset', 'score']].groupby('dataset').agg('max')['score']
        max_performance_original_test = max_performance_original_test[test_datasets]
        # Define Decision Tree features
        train_X = train_metadata.drop('dataset', 1)
        test_X = test_metadata.drop('dataset', 1)
        # Sequence of configuration subsets and Decision Trees
        configuration_subset_sequence = list()
        dt = list()
        # Loop over partition subsets as long as necessary
        keep_loop = 1
        while keep_loop == 1:
            # Create some arrays
            execution_times_sum_partition = np.zeros((len(train_datasets), len(configurations_partition)))
            max_performance_partition = np.zeros((len(train_datasets), len(configurations_partition))) + 0.5
            expected_performance_ratio_partition = np.zeros((len(train_datasets), len(configurations_partition)))
            expected_performance_ratio_partition_test = np.zeros((len(test_datasets), len(configurations_partition)))
            time_differences_sum = np.zeros(len(configurations_partition))
            # Create DT list for this iteration
            dt_iteration = list()
            # Loop over configuration subsets
            for c in range(len(configurations_partition)):
                hp_1 = configurations_partition[c, 0]
                hp_2 = configurations_partition[c, 1]
                train_results_part = train_results.loc[((train_results[hp_1] == 0) | (train_results[hp_2] == 0)), ]
                # Calculate reduced set execution times
                execution_times_fs_part = train_results_part[aggregation_columns].groupby(group_columns).agg('max')
                execution_times_fs_sum_part = execution_times_fs_part.groupby(['dataset']).agg('sum')
                train_datasets_remaining = train_datasets[np.in1d(train_datasets, execution_times_fs_sum_part.index)]
                execution_times_fs_sum_part = execution_times_fs_sum_part.loc[train_datasets_remaining]
                execution_times_ml_sum_part = train_results_part[['dataset', 'time_ml']].groupby(['dataset']).agg('sum')
                execution_times_ml_sum_part = execution_times_ml_sum_part.loc[train_datasets_remaining]
                execution_times_sum_partition[np.in1d(execution_times_sum.index, execution_times_fs_sum_part.index),
                                              c] =\
                    execution_times_fs_sum_part['time_fs'] + execution_times_ml_sum_part['time_ml']
                # Calculate reduced set performance
                max_performance_partition[np.in1d(max_performance.index, execution_times_fs_sum_part.index), c] =\
                    train_results_part[['dataset', 'score']].groupby('dataset').agg('max').loc[train_datasets_remaining,
                                                                                               'score']
                # Calculate DT target variable
                Y_train = max_performance_partition[:, c] / max_performance
                np.random.seed(1234 + c)
                meta_model_ = DecisionTreeRegressor()
                alphas = meta_model_.cost_complexity_pruning_path(train_X, Y_train)
                cv_model = GridSearchCV(meta_model_, cv=5, scoring='neg_mean_squared_error',
                                        param_grid={'min_samples_leaf': [3, 5, 7],
                                                    'ccp_alpha': alphas['ccp_alphas']})
                cv_model.fit(train_X, Y_train)
                meta_model = cv_model.best_estimator_
                meta_model.fit(train_X, Y_train)
                expected_performance_ratio_partition[:, c] = meta_model.predict(train_X)
                expected_performance_ratio_partition_test[:, c] = meta_model.predict(test_X)
                dt_iteration.append({'model': meta_model})
                # Calculate time gains
                time_differences_sum[c] = sum(execution_times_sum[expected_performance_ratio_partition[:, c] >
                                                                  thresh_perf]) -\
                    sum(execution_times_sum_partition[expected_performance_ratio_partition[:, c] > thresh_perf, c])
            print('Min time difference:', round(min(time_differences_sum), 0))
            print('Max time difference:', round(max(time_differences_sum), 0))
            # Update mapping, times, and performances if necessary
            if max(time_differences_sum) > 0:
                # Locate subset
                chosen_subset = np.array(range(len(configurations_partition)))[time_differences_sum ==
                                                                               max(time_differences_sum)]
                chosen_subset = int(chosen_subset[0])
                # Update mapping
                configuration_subset_sequence.append(configurations_partition[chosen_subset])
                dt.append(dt_iteration[chosen_subset])
                subset_to_remove = configurations_partition[chosen_subset]
                for d in range(len(train_metadata)):
                    if expected_performance_ratio_partition[d, chosen_subset] > thresh_perf:
                        mapping[d] = mapping[d][((mapping[d][:, 0] != subset_to_remove[0]) |
                                                 (mapping[d][:, 1] != subset_to_remove[1]))]
                # Update times
                execution_times_sum[expected_performance_ratio_partition[:, chosen_subset] > thresh_perf] =\
                    execution_times_sum_partition[expected_performance_ratio_partition[:, chosen_subset] > thresh_perf,
                                                  chosen_subset]
                print('---')
                out = np.in1d(train_results['dataset'],
                              train_datasets[expected_performance_ratio_partition[:, chosen_subset] >
                                             thresh_perf]) &\
                    (train_results[subset_to_remove[0]] == 1) &\
                    (train_results[subset_to_remove[1]] == 1)
                train_results = train_results.loc[out == 0]
                print('Train dataset updated')
                print('---')
                out_test = np.in1d(test_results['dataset'],
                                   test_datasets[expected_performance_ratio_partition_test[:, chosen_subset] >
                                                 thresh_perf]) &\
                    (test_results[subset_to_remove[0]] == 1) &\
                    (test_results[subset_to_remove[1]] == 1)
                test_results = test_results.loc[out_test == 0]
                print('Test dataset updated')
            else:
                keep_loop = 0
            print('-----------------------------------------')
        # Check start to end differences
        # Calculate execution times
        execution_times_fs_test = test_results[aggregation_columns].groupby(group_columns).agg('max')
        execution_times_fs_sum_test = execution_times_fs_test.groupby(['dataset']).agg('sum')
        execution_times_ml_sum_test = test_results[['dataset', 'time_ml']].groupby(['dataset']).agg('sum')
        execution_times_sum_test = execution_times_fs_sum_test['time_fs'] + execution_times_ml_sum_test['time_ml']
        # Calculate performance
        max_performance = train_results[['dataset', 'score']].groupby('dataset').agg('max')['score']
        max_performance_test = test_results[['dataset', 'score']].groupby('dataset').agg('max')['score']
        # Store results
        performance_ratio_train = max_performance / max_performance_original
        performance_ratio_train.loc[np.isnan(performance_ratio_train)] =\
            0.5 / max_performance_original.loc[np.isnan(performance_ratio_train)]
        results_on_performance_update_vs_not.at[run_, 'perf_ratio'] = np.mean(performance_ratio_train)
        results_on_performance_update_vs_not.at[run_, 'time_ratio'] = (sum(execution_times_sum) /
                                                                       sum(execution_times_sum_original))
        performance_ratio_test = max_performance_test / max_performance_original_test
        performance_ratio_test.loc[np.isnan(performance_ratio_test)] =\
            0.5 / max_performance_original_test.loc[np.isnan(performance_ratio_test)]
        results_on_performance_update_vs_not.at[run_, 'perf_ratio_ho'] = np.mean(performance_ratio_test)
        results_on_performance_update_vs_not.at[run_, 'time_ratio_ho'] = (sum(execution_times_sum_test) /
                                                                          sum(execution_times_sum_original_test))
        print('----- Run', run_)
        print('----- Train -----')
        print('Average performance ratio:', np.round(np.mean(max_performance / max_performance_original), 5))
        print('Ratio of time sum:', round(sum(execution_times_sum) / sum(execution_times_sum_original), 5))
        print('----- Test ------')
        print('Average performance ratio:', np.round(np.mean(max_performance_test / max_performance_original_test), 5))
        print('Ratio of time sum:', round(sum(execution_times_sum_test) / sum(execution_times_sum_original_test), 5))
        print('-----------------------------------------')

    # Save results
    results_on_performance_update_vs_not.to_csv(output_directory + 'MetaTrees_classification_' + str(thresh_perf) +\
                                                '_thresh_' + str(configuration_percentage) + '_configurations_' +\
                                                str(train_percentage) + '_train' + '.csv')
