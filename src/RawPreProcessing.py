import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

class rawpreprocessing:
    def __init__(self, **kwargs):
        self.data_set_per_label = kwargs.get('data_set_per_label', 10)
        self.time_window=kwargs.get('time_window', 3)
        self.num_data_set=10 #set_dataset에서 초기화됨
        self.count=0 # 몇번째 데이터가 들어왔는지 체크
        self.Y_label= None # 행동에 대한 라벨링 값 저장하는 리스트 
        self.raw_array = []  # 3차원 배열을 만들기 위한 리스트
        self.set_data_set(kwargs.get('labels')) # labels를 kwargs로 전달받아 사용


    def set_data_set(self, labels=None):
        if labels is None:
            print("라벨 정보가 없습니다.")
            self.Y_label = None
            self.num_data_set = 0
            return
 
        num_actions = len(labels)
        print(f"[INFO] 세션에서 받은 라벨 목록: {labels}")
        for i, label in enumerate(labels):
            print(f"{i+1}번째 행동의 라벨: {label}")

        self.Y_label = np.array(labels)
        self.num_data_set = num_actions * self.data_set_per_label
        print(f"총 {self.num_data_set}개의 데이터를 처리합니다.")
    
    
    """사용자로부터 행동 개수를 입력받고, 해당 행동에 대한 라벨을 저장"""
    """try:
            num_actions = int(input("몇 가지 행동을 학습하시겠습니까? "))  # 행동 개수 입력 받기
            labels=[]
            for i in range(num_actions):
                action_label = input(f"{i+1}번째 행동의 라벨을 입력하세요: ")  # 행동 라벨 입력 받기
                labels.append(action_label)
                
            # 리스트를 NumPy 배열로 변환하여 저장
            self.Y_label = np.array(labels)

            # 데이터셋 크기를 행동 개수 * data_set_per_label 으로 설정  
            self.num_data_set = num_actions * self.data_set_per_label
            print(f"총 {self.num_data_set}개의 데이터를 처리합니다.")

        except ValueError:
            print("올바른 숫자를 입력해주세요.")"""
        

    def remove_edges_from_csv(self, file_path):
        """서버에서는 사용하지 않음 (클라이언트에서 전처리)"""
        pass

    def filter_data_by_time(self, df, x_value):
        """서버에서는 사용하지 않음 (클라이언트에서 전처리)"""
        pass

    def plot_csv_data(self, df):
        """서버에서는 사용하지 않음 (클라이언트에서 시각화)"""
        pass

    def make_csv_array(self, df):
        """
        CSV 파일을 NumPy 배열로 변환하는 함수.
        """
        try:
            df.rename(columns={
                'Linear Acceleration x (m/s^2)': 'x',
                'Linear Acceleration y (m/s^2)': 'y',
                'Linear Acceleration z (m/s^2)': 'z',
                'Absolute acceleration (m/s^2)': 'a'
            }, inplace=True)
            df.drop(['Time (s)'], axis=1, inplace=True)

            numppy = df.to_numpy()
            #print(f"파일 '{file_name}' 처리 완료!")
            return numppy
        except FileNotFoundError:
            #print(f"파일 '{file_name}'이(가) 존재하지 않습니다.")
            return None
        
    def make_total_array(self):
        """
        raw_array를 3차원 NumPy 배열로 변환하는 함수.
        """
        if self.raw_array:
            total_array = np.stack(self.raw_array, axis=0)
            print("최종 3차원 배열 형태:", total_array.shape)
            return total_array
        else:
            print("읽은 데이터가 없습니다.")