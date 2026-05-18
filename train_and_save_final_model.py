import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from bivariate_filtering import bivariate_filtering
from neural_network import TorchMLPClassifier
from combined_filtering import combined_filtering
from evaluation import evaluate
from hyperparameter_optimization import optimize_hyperparameters, get_param_grids
from normalisation import normalise_train_set
from one_hot_encoding import one_hot_encode
from statistical_analysis import evaluate_models_with_repetitions, statistical_comparison


def filter_dataset(ds):
    initial_count = len(ds)
    print(f"Taille initiale: {initial_count}")

    filtered_ds, st_encoder, wn_area_encoder = one_hot_encode(ds)
    filtered_ds = combined_filtering(filtered_ds)
    filtered_ds = bivariate_filtering(filtered_ds)

    return filtered_ds, st_encoder, wn_area_encoder

def save_model(model, scaler, soil_type_encoder, wilderness_area_encoder, num_equipe="1"):
    joblib.dump(model, f"model_{num_equipe}.pkl")
    joblib.dump(scaler, f"scaler_{num_equipe}.pkl")
    joblib.dump(soil_type_encoder, f"soil_type_{num_equipe}.pkl")
    joblib.dump(wilderness_area_encoder, f"wilderness_{num_equipe}.pkl")

def load_base_dataset(path):
    dataset = pd.read_csv(path)

    dataset = dataset.drop(
        (dataset[(dataset["Cover_Type"] == "Aspen") & (dataset["Horizontal_Distance_To_Fire_Points"] > 3500)]).index)
    dataset = dataset.drop((dataset[(dataset["Cover_Type"] == "Krummholz") & (dataset["Elevation"] < 3000)]).index)
    dataset = dataset.drop((dataset[(dataset["Soil_Type"] == "_9")]).index)
    dataset = dataset.drop((dataset[(dataset["Soil_Type"] == "_17")]).index)
    dataset = dataset.drop((dataset[(dataset["Soil_Type"] == "_38")]).index)
    dataset = dataset.reset_index(drop=True)
    return dataset

def scaled_transform(x_train, x_test):
    x_train_scaled, x_test_scaled, _ = normalise_train_set(x_train, x_test)
    return x_train_scaled, x_test_scaled


def main():
    dataset = load_base_dataset("./data/covertype_project_set.csv")
    print("=== Filtrage ===")

    dataset, st_encoder, wn_area_encoder = filter_dataset(dataset)

    print(f"Taille du jeu de données filtré: {len(dataset)}")

    x = dataset.drop("Cover_Type", axis=1)
    y = dataset["Cover_Type"]

    x_train, x_validation, y_train, y_validation = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("\n=== Répartition des données ===")
    print(f"Taille de l'ensemble d'entrainement: {len(x_train)}")
    print(f"Répartition des classes dans l'ensemble d'entrainement:\n{y_train.value_counts().sort_index()}\n")
    print(f"Taille de l'ensemble de validation: {len(x_validation)}")
    print(f"Répartition des classes dans l'ensemble de validation:\n{y_validation.value_counts().sort_index()}\n")

    x_train_scaled, x_validation_scaled, scaler = normalise_train_set(x_train, x_validation)

    param_grids = get_param_grids()

    nn_params, nn_model = optimize_hyperparameters(
        TorchMLPClassifier(random_state=42),
        param_grids["TorchMLPClassifier"],
        x_train_scaled,
        y_train,
        search_type="random",
    )

    dt_params, dt_model = optimize_hyperparameters(
        DecisionTreeClassifier(random_state=42),
        param_grids["DecisionTreeClassifier"],
        x_train_scaled,
        y_train,
        search_type="random",
    )

    rf_params, rf_model = optimize_hyperparameters(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grids["RandomForestClassifier"],
        x_train_scaled,
        y_train,
        search_type="random",
    )

    knn_params, knn_model = optimize_hyperparameters(
        KNeighborsClassifier(),
        param_grids["KNeighborsClassifier"],
        x_train_scaled,
        y_train,
        search_type="random",
    )

    lr_params, lr_model = optimize_hyperparameters(
        LogisticRegression(random_state=42),
        param_grids["LogisticRegression"],
        x_train_scaled,
        y_train,
        search_type="random",
    )



    print("\n=== Hyperparamètres optimaux ===")
    for title, params in (
        ("Decision Tree", dt_params),
        ("Random Forest", rf_params),
        ("KNN", knn_params),
        ("Logistic Regression", lr_params),
        ("Neural Network", nn_params),
    ):
        print(f"\n{title}:")
        for k, v in sorted(params.items()):
            print(f"  {k}: {v}")

    models_dict = {
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "KNN": knn_model,
        "Logistic Regression": lr_model,
        "Neural Network": nn_model,
    }

    x_transformers_dict = {
        "Decision Tree": scaled_transform,
        "Random Forest": scaled_transform,
        "KNN": scaled_transform,
        "Logistic Regression": scaled_transform,
        "Neural Network": scaled_transform,
    }

    results_df, results_dict, scores_par_repetition = evaluate_models_with_repetitions(
        models_dict,
        x_train,
        y_train,
        x_transformers_dict,
        n_repetitions=10
    )
    print("\n=== Accuracy par modèle et par répétition ===")
    print(scores_par_repetition.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\n=== Accuracy moyenne (répétitions) ===")
    print(results_df)

    print("\n=== IC 95 % (sur les scores des répétitions) ===")
    for name in results_df.index:
        m, marg = results_dict[name]["ci95"]
        print(f"  {name}: {m:.6f} ± {marg:.6f}  →  [{m - marg:.6f}, {m + marg:.6f}]")

    comparison_df = statistical_comparison(results_dict)
    print("\n=== Comparaison statistique (t-test apparié) ===")
    print(comparison_df)

    best_model_name = results_df.index[0]
    best_model = models_dict[best_model_name]

    print(f"\n=== Meilleur modèle ===\n{best_model_name} (accuracy moyenne: {results_df.loc[best_model_name, 'Accuracy']:.6f})")

    best_model.fit(x_train_scaled, y_train)

    save_model(model=best_model,
               scaler=scaler,
               soil_type_encoder=st_encoder,
               wilderness_area_encoder=wn_area_encoder,
               num_equipe="11")

    evaluate(best_model, x_train_scaled, y_train, x_validation_scaled, y_validation)

if __name__ == "__main__":
    main()