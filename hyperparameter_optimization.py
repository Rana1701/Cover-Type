import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier


def optimize_hyperparameters(model, param_grid, x_train, y_train, search_type="grid", cv_folds=5, n_iter=10):
    n_jobs = 1 if isinstance(model, KNeighborsClassifier) else -1
    if search_type == "grid":
        search = GridSearchCV(model, param_grid, cv=cv_folds, n_jobs=n_jobs, verbose=1)
    else:
        search = RandomizedSearchCV(model, param_grid, cv=cv_folds, n_iter=n_iter, n_jobs=n_jobs, random_state=42, verbose=1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module=r"sklearn\.linear_model\..*")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.linear_model\..*")
        warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.linear_model\..*")
        search.fit(x_train, y_train)
    return search.best_params_, search.best_estimator_


def get_param_grids():
    return {
        "DecisionTreeClassifier": {
            "max_depth": [5, 10, 15, 20, 25],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "RandomForestClassifier": {
            "n_estimators": [100, 200, 300],
            "max_depth": [20,30,40,60],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "max_samples": [30000, 50000, None]
        },
        "KNeighborsClassifier": {
            "n_neighbors": [4, 10, 15,20, 30],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]
        },
        "LogisticRegression": {
            "C": [0.001, 0.5, 1.0, 10.0, 20.0, 40.0, 50.0, 100.0, 200],
            "solver": ["lbfgs", "saga"],
            "max_iter": [1000, 3000, 5000],
            "class_weight": [None, "balanced"],
        },
        "TorchMLPClassifier": {
            "hidden_layer_sizes": [
                (256, 128),
                (256, 128, 64),
                (384, 192),
                (512, 256, 128),
            ],
            "optimizer": ["adamw", "adam"],
            "lr": [3e-4, 5e-4, 8e-4, 0.004],
            "batch_size": [256, 512, 1024],
            "max_epochs": [60, 100, 120],
            "dropout": [0.0, 0.1, 0.15, 0.2, 0.25],
            "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
            "patience": [8, 10, 12],
            "scheduler_patience": [2, 3, 4],
            "use_batch_norm": [True, False],
        },
    }
