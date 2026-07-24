from data.data_loader import load_all_subjects

from preprocessing.preprocessor import preprocess_dataset

from features.feature_extractor import (
    extract_features_batch,
    get_feature_dim
)

print("=" * 60)
print("FEATURE EXTRACTION TEST")
print("=" * 60)

# Load dataset
X, y, subjects = load_all_subjects()

print("\nDataset Loaded")
print(X.shape)

# Preprocess
X_clean, y_clean, subjects_clean, pp = preprocess_dataset(
    X,
    y,
    subjects
)

print("\nPreprocessing Finished")
print(X_clean.shape)

# Feature Extraction
features = extract_features_batch(
    X_clean,
    verbose=True
)

print("\nFeature Extraction Completed")
print("=" * 60)

print("Feature Matrix Shape :", features.shape)
print("Expected Feature Dim :", get_feature_dim())

print("One Sample Shape     :", features[0].shape)

print("First 20 Features")
print(features[0][:20])

print("=" * 60)