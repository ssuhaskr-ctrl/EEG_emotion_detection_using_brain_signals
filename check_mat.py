import scipy.io as sio

file = r"C:\GAMEEMO_PROJECT\GAMEEMO\(S01)\Preprocessed EEG Data\.mat format\S01G1AllChannels.mat"

mat = sio.loadmat(file)

print("=" * 60)
print("MAT FILE KEYS")
print("=" * 60)

print(mat.keys())

print()

for key, value in mat.items():

    if key.startswith("__"):
        continue

    print("Variable :", key)
    print("Shape    :", value.shape)
    print("Type     :", value.dtype)
    print()