def get_train_test(data, test_size_weeks=3):

    data_train = data[data['week_no'] < data['week_no'].max() - test_size_weeks]
    data_test = data[data['week_no'] >= data['week_no'].max() - test_size_weeks]

    train_items = data_train['item_id'].unique()
    train_users = data_train['user_id'].unique()

    data_test = data_test[data_test['item_id'].isin(train_items)]
    data_test = data_test[data_test['user_id'].isin(train_users)]

    return data_train, data_test