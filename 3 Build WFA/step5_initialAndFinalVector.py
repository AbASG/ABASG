import numpy as np
import torch
import torch.nn.functional as F

import sys
np.set_printoptions(threshold=sys.maxsize)


# ----- Step 1: Set Hyper Parameters ----- #

CLUSTER_NUM = 452


# ----- Step 2: Load States and Abstract States ----- #

states = np.load("../datasets/states.npy")                          # (2400, 40, 3)
abst_states_labels = np.load("../datasets/abst_states_labels.npy")  # (2400, 40)


# ----- Step 3: Get State_0 ----- #

output_0 = np.zeros(3)
output_0_tensor = torch.from_numpy(output_0)
state_0_tensor = F.softmax(output_0_tensor)
state_0 = state_0_tensor.detach().numpy()

print('--------------------------------------------------------')
print('state_0 shape:\n', state_0.shape)
print('state_0:\n', state_0)


# ----- Step 4: Build Initial Vector ----- #

initial_vector = np.zeros(CLUSTER_NUM+1)
initial_vector[0] = 1

print('--------------------------------------------------------')
print('initial_vector shape:\n', initial_vector.shape)
print('initial_vector:\n', initial_vector)


# ----- Step 5: Build Non-probabilistic Final Vectors ----- #

non_prob_final_vector = np.zeros([CLUSTER_NUM+1, 3])           # (370+1, 3)

for i_0 in range(2400):
    for i_1 in range(40):                                       # for every state
        # state_class = np.argsort(-states[i_0, i_1, :])[:1]
        state_class = np.argsort(-states[i_0, i_1, :])[0]       # the class of state in original problem

        # if i_0 == 0 and i_1 == 0:
        #     print(states[i_0, i_1, :])
        #     print(state_class)

        abst_label = int(abst_states_labels[i_0, i_1])               # the abst label of state

        non_prob_final_vector[abst_label, state_class] += 1     # corresponding class count++

print('--------------------------------------------------------')
print('non_prob_final_vector shape:\n', non_prob_final_vector.shape)
print('non_prob_final_vector:\n', non_prob_final_vector)


# ----- Step 6: Build (Probabilistic) Final Vectors ----- #

final_vector = np.zeros([CLUSTER_NUM+1, 3])                # (370+1, 3)

for item in range(CLUSTER_NUM+1):                           # for every abstract state
    if item == 0:
        final_vector[item, :] = state_0
    else:
        item_classes = non_prob_final_vector[item, :]
        item_sum = np.sum(item_classes)                     # count the num of all the classes
        final_vector[item, :] = item_classes / item_sum     # count the probabilistic of every class

print('--------------------------------------------------------')
print('final_vector shape:\n', final_vector.shape)
print('final_vector:\n', final_vector)
print('final_vector one dim sum:\n', np.sum(final_vector[1, :]))


# ----- Step 7: Save Initial Vector and Final Vectors ----- #

np.save("../datasets/initial_vector.npy", initial_vector)
np.save("../datasets/final_vector.npy", final_vector)
