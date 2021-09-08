import numpy as np

# ----- Step 1: Set Hyper Parameters ----- #

T = 20
K = 3


# ----- Step 2: Load States ----- #

states = np.load("../datasets/states.npy")      # (60000, 28, 10)
print(states[0, 0, :])


# ----- Step 3: Get y and c ----- #

sorted_states = -np.sort(-states)[:, :, :K]          # y (60000, 28, K)
print(sorted_states[0, 0, :])

sorted_states_index = np.argsort(-states)[:, :, :K]  # C (60000, 28, K)
print(sorted_states_index[0, 0, :])


# ----- Step 4: Get n(y(c)) ----- #

sorted_states_t = np.floor(sorted_states*T)
print(sorted_states_t[0, 0, :])


# ----- Step 5: Get k_DCP ((c,n(y(c))), ...) ----- #

k_DCP = np.append(sorted_states_index, sorted_states_t, axis=2)  # k_DCP (2400, 40, 2*K)
print(k_DCP[0, 0, :])


# ----- Step 6: Get P (((c,n(y(c))), ...), ...) ----- #

def findByRow(mat, row):
   return np.where((mat == row).all(1))[0]

abst_states_lables = np.zeros(2400*40).astype(int)  # 2400*40

# optimized
k_DCP = np.reshape(k_DCP, (-1, 2*K))
print(k_DCP[0])
uniques = np.unique(k_DCP, axis=0)
print(len(uniques))

for i in range(len(uniques)):
    j = findByRow(k_DCP, uniques[i, :])
    abst_states_lables[j] = i

abst_states_lables = np.reshape(abst_states_lables, (2400, 40))  # 2400, 40

abst_states_lables = abst_states_lables + 1  # 因为要给不在这里面的0状态留出位置

np.save("../datasets/abst_states_labels.npy", abst_states_lables)


# T   K   len(train)
# 50  3   1787  1994
# 40  3
# 30  3   848   921(18.4)
# 20  3   437   452(11.5)
# 10  3