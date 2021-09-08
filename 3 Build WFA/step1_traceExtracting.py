import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=np.inf)  # print all

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


# ----- Step 1: Load trained Model ----- #

rnn = torch.load('../datasets/rnn_model.pkl')
print(rnn)


# ----- Step 2: Dataset Loading and Preprocessing ----- #

X = np.load("../datasets/X_prep.npy")       # (3000, 40, 2)
Y = np.load("../datasets/Y_prep.npy")       # (3000, 3)

train_size = int(0.8 * X.shape[0])

train_x = X[:train_size, :, :].copy()       # (2400, 40, 2)
train_x = torch.from_numpy(train_x)
train_x = train_x.to(torch.float32)

train_y = Y[:train_size].copy()             # (2400, )


# ----- Step 3: Execute and Verify Model ----- #

train_output, train_output_trace = rnn(train_x)

pred_y = torch.max(train_output, 1)[1].data.numpy()
accuracy = float((pred_y == train_y).astype(int).sum()) / float(train_y.size)
print('train datasets accuracy: %.2f' % accuracy)


# ----- Step 4: Save State ----- #

train_output_trace = F.softmax(train_output_trace, dim=2)  # softmax in last dim
states = train_output_trace.detach().numpy() #

print(states.shape)
print(states[0, :, :])

np.save("../datasets/states.npy", states)


# ----- Step 5: Save Alphabet ----- #

alphabet = train_x.detach().numpy()

print(alphabet.shape)
print(alphabet[0, :, :])

np.save("../datasets/alphabet.npy", alphabet)


# ----- Step 6: Save Triples ----- #


