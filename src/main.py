from train_model import train_m
from test_model import test_m
import numpy as np

data_set=np.load('train_data.npy')
num=len(data_set)
Y_label=['handshaking', 'punching', 'waving', 'walking', 'running']

train_m(data_set=data_set, Y_label=Y_label)

test=np.load('test_data.npy')
test_m(test=test, Y_label=Y_label)