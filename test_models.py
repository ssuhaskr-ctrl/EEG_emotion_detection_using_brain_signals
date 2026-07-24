import numpy as np

from models.classical_models import ClassicalModels

print("=" * 60)
print("GAMEEMO CLASSICAL MODELS TEST")
print("=" * 60)

# Generate dummy feature matrix
X = np.random.randn(1000, 150).astype(np.float32)

# Generate 4-class labels
y = np.random.randint(0, 4, 1000)

# Create model object
model = ClassicalModels()

# Train all models
scores = model.train(X, y)

# Save best model
model.save_best_model()

# Load best model
best = model.load_model()

print()
print("=" * 60)
print("Loaded Best Model Successfully")
print(type(best))

# Test prediction
pred = best.predict(X[:10])

print()
print("Prediction Shape :", pred.shape)
print("Predictions      :", pred)

print()
print("TEST COMPLETED SUCCESSFULLY")
print("=" * 60)