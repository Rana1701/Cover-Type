import pandas as pd
from sklearn.preprocessing import OneHotEncoder

features = ['Wilderness_Area', 'Soil_Type']

def one_hot_encode(dataset):
    soil_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    soil_type_encoder.fit(dataset[['Soil_Type']])

    wilderness_area_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    wilderness_area_encoder.fit(dataset[['Wilderness_Area']])

    dataset_filled = dataset.copy()

    df = pd.get_dummies(dataset_filled, columns=features, dtype=int)
    return df, soil_type_encoder, wilderness_area_encoder

if __name__ == "__main__":
    ds = pd.read_csv("./data/covertype_project_set.csv")
    one_hot_encode(ds)
