"""
test.py
Test the GAMEEMO data loading and preprocessing pipeline.
"""

from data.data_loader import load_all_subjects
from preprocessing.preprocessor import preprocess_dataset


def main():

    print("=" * 60)
    print("GAMEEMO PIPELINE TEST")
    print("=" * 60)

    # Load dataset
    X, y, subjects = load_all_subjects(verbose=True)

    print("\nDataset Loaded Successfully")
    print(f"X Shape       : {X.shape}")
    print(f"y Shape       : {y.shape}")
    print(f"Subjects Shape: {subjects.shape}")

    # Preprocess dataset
    X_clean, y_clean, subjects_clean, preprocessor = preprocess_dataset(
        X,
        y,
        subjects,
        apply_ica=False
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(f"Processed X        : {X_clean.shape}")
    print(f"Processed y        : {y_clean.shape}")
    print(f"Processed Subjects : {subjects_clean.shape}")

    print("\nClass Distribution:")

    for cls in sorted(set(y_clean)):
        print(f"Class {cls} -> {(y_clean == cls).sum()} samples")

    print("\nPipeline test completed successfully.")


if __name__ == "__main__":
    main()