import pandas as pd

def bivariate_filtering(dataset):

    df = dataset[
        (dataset["Elevation"] >= 1900) & (dataset["Elevation"] <= 3850)]
    mask_bad = (df["Elevation"] < 2200) & (df["Horizontal_Distance_To_Fire_Points"] > 3000)
    df = df[~mask_bad]
    mask_aspen = (df["Cover_Type"] == "Aspen") & (df["Horizontal_Distance_To_Fire_Points"] > 3400)
    df = df[~mask_aspen]
    mask_krummholz = (df["Cover_Type"] == "Krummholz") & (df["Elevation"] < 3050)
    df = df[~mask_krummholz]

    label = "Cover_Type"
    numeric_cols = df.select_dtypes(include="number").columns.difference([label])
    if len(numeric_cols) > 0:
        lower = df[numeric_cols].quantile(0.005)
        upper = df[numeric_cols].quantile(0.995)
        within = ((df[numeric_cols] >= lower) & (df[numeric_cols] <= upper)).all(axis=1)
        df = df[within]

    return df

if __name__ == "__main__":
    ds = pd.read_csv("./data/filtered/covertype_project_set_filtered.csv")
    bivariate_filtering(ds)
