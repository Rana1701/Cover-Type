from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def compute_metrics(y_true, y_pred):
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0)
    }


def evaluate(model, x_train, y_train, x_test=None, y_test=None, print_matrices=True):

    if x_test is None or y_test is None:
        y_full_pred = model.predict(x_train)
        metrics = compute_metrics(y_train, y_full_pred)

        print("Evaluation on full dataset:")
        if print_matrices:
            print("Confusion Matrix:")
            print(metrics["confusion_matrix"])
        print("Accuracy:")
        print(metrics["accuracy"])
        print("Precision weighted:")
        print(metrics["precision_weighted"])
        print("Recall weighted:")
        print(metrics["recall_weighted"])
        print("\n")
        return metrics

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    if print_matrices:
        print("Confusion Matrix in training:")
        print(train_metrics["confusion_matrix"])
        print("Confusion Matrix in test:")
        print(test_metrics["confusion_matrix"])
        print("\n")

    print("Accuracy in training:")
    print(train_metrics["accuracy"])
    print("Precision weighted in training:")
    print(train_metrics["precision_weighted"])
    print("Recall weighted in training:")
    print(train_metrics["recall_weighted"])

    print("Accuracy in test:")
    print(test_metrics["accuracy"])
    print("Precision weighted in test:")
    print(test_metrics["precision_weighted"])
    print("Recall weighted in test:")
    print(test_metrics["recall_weighted"])
    print("\n")

    return {
        "train": train_metrics,
        "test": test_metrics
    }