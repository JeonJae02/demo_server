from model import GRUMotionClassifier
import Data_Extract
from SlidingWindow import slidingwindow
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def train_m(data_set, Y_label, stat_variable=103, fft_variable=1, callback=None):
    num=len(data_set)

    X=[]
    y=[]
    sliding_window_processor = slidingwindow(data_set, Y_label)
    for j in range(0, num):  # row data 갯수 만큼 돌림
            part_data = data_set[j]

            # Fourier 변환을 통해 최대 주파수 구하기
            max_freq = sliding_window_processor.fourier_trans_max_amp(part_data[:, 3], 100)  # absolute 값
            #print(f"Max Frequency for dataset {j}: {max_freq}")

            # SlidingWindow 클래스 인스턴스 생성 및 슬라이딩 윈도우 처리
            win_datas=sliding_window_processor.sliding_window(1/max_freq,1/max_freq*0.5,j)
            #print(Data_Extract.data_extraction(win_datas[3]).extract_feature())
            for i in range(0, len(win_datas)):
                    
                    X.append(Data_Extract.data_extraction(win_datas[i], stat_variable=stat_variable, fft_variable=fft_variable).extract_feature())
                    y.append(Y_label[int(j/10)])


    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)   # (batch_size, seq_length, input_dim)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)   # (batch_size, seq_length, input_dim)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Dataset & DataLoader 설정
    # dateset을 n개로 나눠서 최적화 진행
    batch_size = 32
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # 시계열 데이터니깐 shuffle을 하면 안되지만 sliding window를 사용했기때문에 여기선 True
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = GRUMotionClassifier(input_size=len(X[1]), hidden_size=64, num_layers=2, output_size=len(Y_label))
    # input_size는 현재 x, y, z, a에서 뽑은 feature 40개
    # hidden_size는 이전 데이터를 얼마나 기억할 것인지, 높으면 정확성이 올라가지만 너무 올라가면 과적합
    # num_layers는 GRU 층
    # output_size는 y_label의 개수


    # ========== 3. 학습 설정 ==========
    learning_rate=0.001 # 학습률, weight를 update할때 얼만큼 weight를 조정할건지, 너무 크면 확확 바뀌고, 너무 작으면 찔끔찔끔 변화함. 적당한 것이 0.001
    criterion = nn.CrossEntropyLoss()  # 분류 문제 -> CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 처음에는 큰 lr을 사용하다가 점차 작은 lr을 사용하는 최적화 알고리즘


    # ========== 4. 학습 실행 ==========
    num_epochs = 60

    for epoch in range(num_epochs):
        model.train()  # 모델을 훈련 모드로 설정
        for batch_X, batch_y in data_loader:
            optimizer.zero_grad() # 이전 Epoch에서 계산된 기울기(Gradient) 초기화
        
            # Forward
            outputs = model(batch_X) # 모델에 입력 데이터를 넣어 예측값 계산
            loss = criterion(outputs, batch_y)  # 손실(loss) 계산

            # Backward & Optimize
            loss.backward() # 역전파(Backpropagation) 수행하여 기울기 계산
            optimizer.step() # 가중치 업데이트
        
        model.eval()  # 모델을 평가 모드로 설정
        total_val_loss = 0
        with torch.no_grad():  # 검증 시에는 gradient 계산을 하지 않음
            for val_X, val_y in val_loader:  # 검증 데이터셋에 대해 예측
                val_outputs = model(val_X)
                val_loss = criterion(val_outputs, val_y)  # 검증 손실 계산
                total_val_loss += val_loss.item()  # 누적 검증 손실 계산

        avg_val_loss = total_val_loss / len(val_loader)  # 평균 검증 손실 계산

        # 10번마다 훈련 손실 및 검증 손실 출력
        if (epoch + 1) % 10 == 0: 
            message = (f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f} 진행중")
            print(message)
            if callback:
                callback(message)
            else:
                 print(message)

    return model, label_encoder