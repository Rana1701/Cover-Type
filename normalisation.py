from sklearn.preprocessing import MinMaxScaler

def normalise_train_set(x_train, x_test):
    scaler = MinMaxScaler()

    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()

    numeric_columns = x_train.select_dtypes(include=["int64", "float64"]).columns

    x_train_scaled[numeric_columns] = scaler.fit_transform(x_train[numeric_columns])
    x_test_scaled[numeric_columns] = scaler.transform(x_test[numeric_columns])

    return x_train_scaled, x_test_scaled, scaler