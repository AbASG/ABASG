import torch
from torch import nn
import numpy as np

torch.manual_seed(1)                      # reproducible
torch.set_printoptions(threshold=np.inf)  # print all


# ----- Step 1: Set Hyper Parameters ----- #

EPOCH = 20
BATCH_SIZE = 100
TIME_STEP = 40
INPUT_SIZE = 2
LR = 0.001


# ----- Step 2: Dataset Loading and Preprocessing ----- #

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


X = np.load("../datasets/X_prep.npy")       # (3000, 40, 2)
Y = np.load("../datasets/Y_prep.npy")       # (3000, 3)

train_size = int(0.8 * X.shape[0])

train_x = X[:train_size, :, :]                          # (2400, 40, 2)
train_x = torch.from_numpy(train_x).to(torch.float32)
#train_x = train_x.to(torch.float32)

train_y = Y[:train_size]                                # (2400, )
train_y = torch.from_numpy(train_y).to(torch.long)

train_data = GetLoader(train_x, train_y)                # 返回Dataset对象(包含data和label)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_x = X[train_size:, :, :]                           # (600, 40, 2)
test_x = torch.from_numpy(test_x).to(torch.float32)

test_y = Y[train_size:]                                 # (600, )

# ----- Step 3: Create Model Class ----- #

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


# ----- Step 4: Instantiate ----- #

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)

loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted


# ----- Step 5: Model Training ----- #

for epoch in range(EPOCH):
    for step, (train_x, train_y) in enumerate(train_loader):
        train_x = train_x.view(-1, TIME_STEP, INPUT_SIZE)

        output, _ = rnn(train_x)
        loss = loss_func(output, train_y)

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()        # backward and compute gradients
        optimizer.step()       # apply gradients

        if step % 50 == 0:
            test_output, test_output_trace = rnn(test_x)
            # print(test_output_trace.size())
            # print(test_output_trace)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

torch.save(rnn, '../datasets/rnn_model.pkl')
