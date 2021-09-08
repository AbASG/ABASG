import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)


# ----- Step 1: Set Hyper Parameters ----- #

CLUSTER_NUM = 452
TOKEN_CLUSTER_NUM = 2004


# ----- Step 2: Load Abstract States and Abstract Alphabet ----- #

abst_states_labels = np.load("../datasets/abst_states_labels.npy")      # (2400, 40)
abst_alphabet_labels = np.load("../datasets/abst_alphabet_labels.npy")  # (2400, 40)


# ----- Step 3: Build Non-probabilistic Transition Matrixes ----- #

non_prob_transition_matrixes = np.zeros((TOKEN_CLUSTER_NUM, CLUSTER_NUM+1, CLUSTER_NUM+1))  # (59, 370+1, 370+1)

for item in range(TOKEN_CLUSTER_NUM):                               # for every abstract token
    for i_0 in range(2400):
        for i_1 in range(40):                                       # for every token
            if abst_alphabet_labels[i_0, i_1] == item:
                if i_1 == 0:                                        # confirm front abstract state
                    front_abst_state = 0
                else:
                    front_abst_state = int(abst_states_labels[i_0, i_1-1])

                back_abst_state = int(abst_states_labels[i_0, i_1])      # confirm back abstract state

                non_prob_transition_matrixes[item, front_abst_state, back_abst_state] += 1  # add a transition edge

print('--------------------------------------------------------')
print('non_prob_transition_matrixes shape:\n', non_prob_transition_matrixes.shape)
print('non_prob_transition_matrixes last dim:\n', non_prob_transition_matrixes[0, 0, :])


# ----- Step 4: Build Probabilistic Transition Matrixes ----- #

prob_transition_matrixes = np.zeros((TOKEN_CLUSTER_NUM, CLUSTER_NUM+1, CLUSTER_NUM+1))  # (59, 370+1, 370+1)

for item in range(TOKEN_CLUSTER_NUM):                               # for every abstract token
    item_matrix = non_prob_transition_matrixes[item, :, :]
    item_sum = np.sum(item_matrix)                                  # count the num of all the transition edges
    prob_transition_matrixes[item, :, :] = item_matrix / item_sum   # count the probabilistic of every transition edges

print('--------------------------------------------------------')
print('prob_transition_matrixes shape:\n', prob_transition_matrixes.shape)
print('prob_transition_matrixes last dim:\n', prob_transition_matrixes[0, 0, :])
print('prob_transition_matrixes one transition matrix sum:\n', np.sum(prob_transition_matrixes[1, :, :]))


# ----- Step 5: Save Probabilistic Transition Matrixes ----- #

np.save("../datasets/prob_transition_matrixes.npy", prob_transition_matrixes)


# 921  791   (45.7)
# 452  1306  (47.8/69.6)
