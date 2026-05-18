from imblearn.over_sampling import RandomOverSampler

def oversample_train_set(x_train, y_train):
    print(f"Initial dataset size: {len(x_train)} rows")
    sampler = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    print(f"Resampled dataset size: {len(x_resampled)} rows")
    return x_resampled, y_resampled