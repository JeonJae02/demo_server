import torch
import torch.nn as nn

class GRUMotionClassifier(nn.Module):
    def __init__(self, input_size=0, hidden_size=64, num_layers=2, output_size=0):
        super(GRUMotionClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size) # 분류 문제이기 때문에 outputsize를 받음. 만약 regression문제라면 1로 고정정

    def forward(self, x):
        gru_out, hidden = self.gru(x)  # hidden: (num_layers, batch_size, hidden_size)
        # 만약 gru_out이 2차원 텐서라면, 시퀀스 길이가 1인 경우일 수 있음
        if gru_out.ndimension() == 2:  # (batch_size, hidden_size)일 때
            out = self.fc(gru_out)  # 바로 fc 레이어에 전달
        else:  # (batch_size, seq_length, hidden_size)일 때
            out = self.fc(gru_out[:, -1, :])  # 마지막 시점의 hidden state 사용
            
        return out

class RNNMotionClassifier(nn.Module):
    def __init__(self, input_size=0, hidden_size=64, num_layers=2, output_size=0):
        super(RNNMotionClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        if rnn_out.ndimension() == 2:
            out = self.fc(rnn_out)
        else:
            out = self.fc(rnn_out[:, -1, :])
        return out

"""class CNNMotionClassifier(nn.Module):
    def __init__(self, input_channels=1, num_classes=0): #input channel 시계열은 1, 
        super(CNNMotionClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(64 * (input_size // 2), num_classes)  # input_size는 시퀀스 길이

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x"""