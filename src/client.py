import requests
import numpy as np
import pandas as pd

# 서버 URL (Flask 서버가 실행 중이어야 합니다)
BASE_URL = "http://127.0.0.1:5000"

# 세션을 유지하기 위해 requests.Session 사용
session = requests.Session()

# 1. 서버에 초기화 요청 (client_id 생성)
initialize_response = session.get("http://127.0.0.1:5000/initialize")
print(initialize_response.json())

print("학습을 시작합니다.")
num_input = input("csv파일:1, npy파일:2")

if(num_input == 1):
    exit() #구현전
else:
    dataset=np.load('train_data.npy')
    
    data_set=dataset.tolist()
    Y_label=['handshaking', 'punching', 'waving', 'walking', 'running']
    """num_y=input("y 몇 개?")
    for i in range(0, num_y):
        tmp = input("{i+1}번째 label")
        Y_label.append(tmp)"""
    
    url = f"{BASE_URL}/input_npy_data"
    json={"data_set": data_set, "Y_label":Y_label}
    response = session.post(url, json=json)
    print("== Get Specific Data ==")
    print(response.json())

#choose_model=input("이용할 모델 골라")
#구현전


print("학습을 시작")
url = f"{BASE_URL}/train_data"
response = session.post(url)
print("== Training Data ==")
print(response.json())


print("test 데이터 갖고오셈")
np_test=np.load('test_data.npy')
test=np_test.tolist()

print("테스트")
url = f"{BASE_URL}/test"
response = session.post(url, json={"test": test})
print("== Testing Data ==")
print(response.json())


print("\n클라이언트: 세션 초기화")
url = f"{BASE_URL}/clear"
response = session.post(url)
print("== Clear Session ==")
print(response.json())