from data.data_loader import load_all_subjects

X, y, sids = load_all_subjects()

print(X.shape)
print(y.shape)
print(sids.shape)