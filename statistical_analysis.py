from scipy import stats
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from dataset_split import create_fixed_split


def evaluate_models_with_repetitions(models_dict, full_x, full_y, x_trains_transformers, n_repetitions=10):
    results = {}
    scores_dict = {model_name: [] for model_name in models_dict.keys()}

    dataset = pd.concat([full_x, full_y.rename("Cover_Type")], axis=1)

    for repetition in range(n_repetitions):
        x_train, x_test, y_train, y_test = create_fixed_split(
            dataset.sample(frac=1, random_state=42 + repetition).reset_index(drop=True),
            label_column="Cover_Type",
            test_samples_per_class=500,
            random_state=repetition * 42,
        )

        for model_name, model in models_dict.items():
            current_model = clone(model)
            transform_fn = x_trains_transformers[model_name]
            x_train_transformed, x_test_transformed = transform_fn(x_train, x_test)

            current_model.fit(x_train_transformed, y_train)
            predictions = current_model.predict(x_test_transformed)
            score = accuracy_score(y_test, predictions)
            scores_dict[model_name].append(score)

    for model_name, scores in scores_dict.items():
        results[model_name] = np.mean(scores)

    results_df = pd.DataFrame({
        "Accuracy": results
    }).sort_values("Accuracy", ascending=False)

    scores_summary = {
        model_name: {
            "scores": scores,
            "mean": np.mean(scores),
            "ci95": mean_confidence_interval(scores)
        }
        for model_name, scores in scores_dict.items()
    }

    repetitions_df = pd.DataFrame(scores_dict)
    repetitions_df.index.name = "Répétition"

    return results_df, scores_summary, repetitions_df


def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data, dtype=float)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)
    return mean, margin


def statistical_comparison(results_dict, alpha=0.05):
    model_names = list(results_dict.keys())
    comparison_results = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1_name = model_names[i]
            model2_name = model_names[j]

            scores1 = results_dict[model1_name]["scores"]
            scores2 = results_dict[model2_name]["scores"]

            t_stat, p_value = stats.ttest_rel(scores1, scores2)

            comparison_results.append({
                "Model 1": model1_name,
                "Model 2": model2_name,
                "T-Statistic": round(t_stat, 4),
                "P-Value": round(p_value, 4),
                "Significant": "Yes" if p_value < alpha else "No"
            })

    return pd.DataFrame(comparison_results)