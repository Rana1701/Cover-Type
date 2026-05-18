import pandas as pd

FILTERS = [
    ("elevation", lambda df: (df["Elevation"] > 2000) & (df["Elevation"] < 3500)),
    ("slope", lambda df: (df["Slope"] < 45)),
    ("horizontal_distance_to_hydrology", lambda df: (df["Horizontal_Distance_To_Hydrology"] < 1200)),
    ("vertical_distance_to_hydrology", lambda df: (df["Vertical_Distance_To_Hydrology"] > -50) & (df["Vertical_Distance_To_Hydrology"] < 250)),
    ("horizontal_distance_to_roadways", lambda df: (df["Horizontal_Distance_To_Roadways"] < 6500)),
    ("horizontal_distance_to_fire_points", lambda df: (df["Horizontal_Distance_To_Fire_Points"] < 6500)),
    ("hillshade_9am", lambda df: (df["Hillshade_9am"] > 100)),
    ("hillshade_noon", lambda df: (df["Hillshade_Noon"] > 140)),
    ("hillshade_3pm", lambda df: (df["Hillshade_3pm"] > 50) & (df["Hillshade_3pm"] < 240)),
]

def combined_filtering(dataset):
    combined_mask = pd.Series(True, index=dataset.index)
    for suffix, mask_fn in FILTERS:
        combined_mask &= mask_fn(dataset)
    soil_cols = [col for col in dataset.columns if col.startswith('Soil_Type__')]
    soil_counts = {col: (dataset[col] == 1).sum() for col in soil_cols}
    valid_soils = [col for col, count in soil_counts.items() if count >= 100]
    if valid_soils:
        soil_mask = dataset[valid_soils].any(axis=1)
        combined_mask &= soil_mask
    df_filtered_combined = dataset[combined_mask].copy()
    return df_filtered_combined

if __name__ == "__main__":
    dataset = pd.read_csv("./data/filtered/covertype_project_set_filtered.csv")
    combined_filtering(dataset)
