import torch
from torch import nn
import numpy as np
import time


# import
INPUT_SIZE = 2
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.out = nn.Linear(128, 3)

    def forward(self, x):
        # x            (batch, time_step, input_size)
        # r_out        (batch, time_step, output_size)
        # h_n and h_c  (n_layers, batch, hidden_size)

        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        out = self.out(r_out[:, -1, :])        # choose r_out at the last time step

        out_trace = self.out(r_out)            # choose r_out at all time steps

        return out, out_trace
class GetLoader(torch.utils.data.Dataset):      # 定义GetLoader类，继承Dataset方法

    def __init__(self, data_root, data_label):  # 初始化，加载数据
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):               # index是根据batchsize划分数据得到的索引
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):                          # 返回数据大小长度，方便DataLoader划分
        return len(self.data)


# ----- Step 1: Load MNIST train Dataset ----- #

X = np.load("../datasets/X_prep.npy")       # (3000, 40, 2)
Y = np.load("../datasets/Y_prep.npy")       # (3000, 3)

train_size = int(0.8 * X.shape[0])

train_x = X[:train_size, :, :].copy()       # (2400, 40, 2)
train_x = torch.from_numpy(train_x)
train_x = train_x.to(torch.float32)

train_y = Y[:train_size].copy()             # (2400, )


# ----- Step 2-1: Load Model ----- #

rnn = torch.load('../datasets/rnn_model.pkl')
print(rnn)


# ----- Step 2-2: Execute Model ----- #

time_start = time.time()                                            # time start
Model_output, _ = rnn(train_x)
time_end = time.time()                                              # time end

print('Model - Run time: %fs' % (time_end - time_start))

Model_pred_y = torch.max(Model_output, 1)[1].data.numpy()
accuracy = float((Model_pred_y == train_y).astype(int).sum()) / float(train_y.size)

print('Model - Accuracy: %.2f' % accuracy)


# ----- Step 3-1: Load WFA (5 components) ----- #

states = np.load("../datasets/states.npy")                                      # (60000, 28, 10)
abst_states_labels = np.load("../datasets/abst_states_labels.npy")              # (60000, 28)

alphabet = np.load("../datasets/alphabet.npy")                                  # (60000, 28, 28)
abst_alphabet_labels = np.load("../datasets/abst_alphabet_labels.npy")          # (60000, 28)

initial_vector = np.load("../datasets/initial_vector.npy")                      # (726+1)

prob_transition_matrixes = np.load("../datasets/prob_transition_matrixes.npy")  # (100,726+1,726+1)

final_vector = np.load("../datasets/final_vector.npy")                          # (726+1,10)


# ----- Step 3-2: Execute WFA ----- #

WFA_output = np.zeros([2400, 3])

time_count = 0.0
total_time_start = time.time()                                  # time start

for i_0 in range(2400):
    output = initial_vector

    for i_1 in range(40):
        index = int(abst_alphabet_labels[i_0, i_1])                  # confirm the abst label of the token case
        transition_matrix = prob_transition_matrixes[index, :, :]

        time_start = time.time()                                # time start
        output = np.matmul(output, transition_matrix)
        time_end = time.time()                                  # time end
        time_count += time_end - time_start

    time_start = time.time()                                    # time start
    output = np.matmul(output, final_vector)
    time_end = time.time()                                      # time end
    time_count += time_end - time_start

    WFA_output[i_0, :] = output

total_time_end = time.time()                                    # time end

print('WFA - Total runtime: %fs' % (total_time_end - total_time_start))
print('WFA - Accurate runtime: %fs' % time_count)

WFA_output = torch.from_numpy(WFA_output)                       # transform to tensor
WFA_pred_y = torch.max(WFA_output, 1)[1].data.numpy()
accuracy = float((WFA_pred_y == train_y).astype(int).sum()) / float(train_y.size)

print('WFA - Accuracy: %.2f' % accuracy)


# a = np.zeros(325)
# b = np.zeros(326)
# c = prob_transition_matrixes[0, :, :]
# print(c)
# print(c.shape)
#
# c = np.insert(c, 0, values=a, axis=0)
# c = np.insert(c, 0, values=b, axis=1)
# print(c)
# print(c.shape)
#
# cyz = np.matmul(initial_vector, c)
# print(cyz)




