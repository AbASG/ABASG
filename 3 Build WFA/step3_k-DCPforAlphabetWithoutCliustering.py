import numpy as np

# ----- Step 1: Set Hyper Parameters ----- #

T = 50


# ----- Step 2: Load States ----- #

alphabet = np.load("../datasets/alphabet.npy")      # (2400, 40, 2)

# ----- Step 3: Get "k-DCP" ----- #

k_DCP = np.floor(alphabet*T)
print(k_DCP[0, 0, :])


# ----- Step 4: Get P ----- #

def findByRow(mat, row):
   return np.where((mat == row).all(1))[0]

abst_alphabet_lables = np.zeros(2400*40).astype(int)  # 2400*40

k_DCP = np.reshape(k_DCP, (-1, alphabet.shape[2]))
uniques = np.unique(k_DCP, axis=0)
print(len(uniques))

for i in range(len(uniques)):
    j = findByRow(k_DCP, uniques[i, :])
    abst_alphabet_lables[j] = i

abst_alphabet_lables = np.reshape(abst_alphabet_lables, (2400, 40))  # 2400, 40

np.save("../datasets/abst_alphabet_labels.npy", abst_alphabet_lables)


# T   K   len(train)
# 400 2
# 200 2   424
# 150 2
# 100 2         6075
# 50  2         1905
# 40  2         1306(32.6)  1386
# 30  2         791(22.7)
# 20  2