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