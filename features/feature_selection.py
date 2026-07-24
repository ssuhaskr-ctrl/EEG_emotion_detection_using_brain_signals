from data.data_loader import load_all_subjects
from preprocessing.preprocessor import preprocess_dataset
from features.feature_extractor import extract_features_batch
from features.feature_selection import (
    select_features,
    print_feature_info
)

print("=" * 60)
print("FEATURE SELECTION TEST")
print("=" * 60)

# Load dataset
X, y, subjects = load_all_subjects()

print("\nDataset Loaded")
print(X.shape)

# Preprocess
X, y, subjects, _ = preprocess_dataset(X, y, subjects)

print("\nPreprocessing Finished")
print(X.shape)

# Extract features
features = extract_features_batch(X)

print("\nFeature Extraction Finished")
print(features.shape)

# Select best features
selected_features, selector = select_features(
    features,
    y,
    k_features=150
)

print_feature_info(features, selected_features)

selector.save()