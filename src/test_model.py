import torch
from model import GRUMotionClassifier
from SlidingWindow import slidingwindow
import numpy as np
import Data_Extract
import joblib  # LabelEncoder 불러오기

def test_m(test, model, label_encoder, Y_label, stat_variable=103, fft_variable=1):
    tests=[]

    sliding_window_test = slidingwindow(test, Y_label)
    for j in range(0, len(test)):  # row data 갯수 만큼 돌림
            part_data = test[j]

            # Fourier 변환을 통해 최대 주파수 구하기
            max_freq = sliding_window_test.fourier_trans_max_amp(part_data[:, 3], 100)  # absolute 값
            #print(f"Max Frequency for dataset {j}: {max_freq}")

            # SlidingWindow 클래스 인스턴스 생성 및 슬라이딩 윈도우 처리
            win_datas=sliding_window_test.sliding_window(1/max_freq,1/max_freq*0.5,j)
            tests.append(Data_Extract.data_extraction(win_datas[len(win_datas)//2], stat_variable=stat_variable, fft_variable=fft_variable).extract_feature())

    test_sample = torch.tensor(tests, dtype=torch.float32)


    # ========== 5. 테스트 ==========
    model.eval()
    with torch.no_grad():
        prediction = model(test_sample)
        predicted_class = torch.argmax(prediction, dim=1)
    for i, pred in enumerate(predicted_class):
        print(f"Test Sample {i+1}: Predicted Motion = {label_encoder.inverse_transform([pred.item()])}")
    # 예측값과 실제값을 비교하여 출력

    return predicted_class