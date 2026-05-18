import joblib
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import accuracy_score


class ModelEvaluator:
    def __init__(self, verbose=False):
        self.soil_type_encoder = None
        self.wilderness_area_encoder = None
        self.scaler = None
        self.model = None
        # Vos features doivent respecter cet ordre dans votre modéle
        self.features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
                    'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
        self.cat_features = ['Wilderness_Area', 'Soil_Type']
        self.label = "Cover_Type"
        self.verbose = verbose

    @staticmethod
    def get_team_number():
        return "11"

    @staticmethod
    def get_members_names():
        return "William Blanchet Lafrenière et Rana Azemdroub"

    @staticmethod
    def compute_metrics(true_labels, prediction, metric, factor=1.0):
        value = factor * metric(true_labels, prediction)
        return value

    @staticmethod
    def mean_confidence_interval(data, confiance=0.95):
        moyenne, ecart_type = np.mean(data), scipy.stats.sem(data)
        intervalle = ecart_type * scipy.stats.t.ppf((1 + confiance) / 2., len(data) - 1)
        return f"{round(moyenne, 2)} Â± {round(intervalle, 2)}"

    def predict(self, x):
        self.load_models()
        x_encoded_wilderness = pd.DataFrame(self.wilderness_area_encoder.transform(x[["Wilderness_Area"]]), columns=self.wilderness_area_encoder.get_feature_names_out(), index=x.index)
        x_encoded_soil_type = pd.DataFrame(self.soil_type_encoder.transform(x[["Soil_Type"]]), columns=self.soil_type_encoder.get_feature_names_out(), index=x.index)
        x = x.drop(columns=["Wilderness_Area", "Soil_Type"])
        x_final = pd.concat([x, x_encoded_wilderness, x_encoded_soil_type], axis=1)
        x_final_scaled = pd.DataFrame(self.scaler.transform(x_final), columns=x_final.columns)

        return self.model.predict(x_final_scaled)

    def score(self, path_do_dataset):
        dataset = self.load_and_filter_dataset(path_do_dataset)
        accuracies_per_repetition = []
        for repetition in range(10):
            sub_sample = dataset.groupby(self.label, group_keys=False).sample(n=500, random_state=repetition * 42)
            predictions = self.predict(sub_sample.drop(columns=[self.label]))
            accuracy = self.compute_metrics(sub_sample[self.label], predictions,
                                       accuracy_score,
                                       factor=100.0)
            accuracies_per_repetition.append(accuracy)
            if self.verbose:
                print(repetition, ":", accuracy)
        return self.mean_confidence_interval(accuracies_per_repetition), accuracies_per_repetition

    def load_and_filter_dataset(self, path_to_dataset):
        dataset = pd.read_csv(path_to_dataset)
        features = self.features
        features.extend(self.cat_features)
        features.append(self.label)
        dataset = dataset[features]
        dataset = dataset.drop((dataset[
            (dataset["Cover_Type"] == "Aspen") & (dataset["Horizontal_Distance_To_Fire_Points"] > 3500)]).index)
        dataset = dataset.drop((dataset[(dataset["Cover_Type"] == "Krummholz") & (dataset["Elevation"] < 3000)]).index)
        dataset = dataset.drop((dataset[(dataset["Soil_Type"] == "_9")]).index)
        dataset = dataset.drop((dataset[(dataset["Soil_Type"] == "_17")]).index)
        dataset = dataset.drop((dataset[(dataset["Soil_Type"] == "_38")]).index)
        dataset = dataset.reset_index(drop=True)
        return dataset

    def load_models(self):
        self.model = joblib.load(f"model_{self.get_team_number()}.pkl")
        self.scaler = joblib.load(f"scaler_{self.get_team_number()}.pkl")
        self.soil_type_encoder = joblib.load(f"soil_type_{self.get_team_number()}.pkl")
        self.wilderness_area_encoder = joblib.load(f"wilderness_{self.get_team_number()}.pkl")




model_to_test = ModelEvaluator()
print(model_to_test.get_members_names(), model_to_test.get_team_number(), model_to_test.score("../../covertype_project_evaluation_set.csv"), sep=",")
