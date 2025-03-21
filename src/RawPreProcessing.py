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
        self.set_data_set()

    def set_data_set(self):
        """사용자로부터 행동 개수를 입력받고, 해당 행동에 대한 라벨을 저장"""
        try:
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
            print("올바른 숫자를 입력해주세요.")
        

    def remove_edges_from_csv(self, file_path):
        """
        CSV 파일에서 앞뒤 10% 데이터를 제거하는 함수.
        """
        df = pd.read_csv(file_path)
        total_rows = len(df)
        ten_percent = int(total_rows * 0.1)
        df_trimmed = df.iloc[ten_percent:-ten_percent]
        return df_trimmed

    def filter_data_by_time(self, df, x_value):
        """
        특정 X 값부터 time_window(기본 3초) 내의 데이터를 필터링하는 함수.
        """
        x_column = df.iloc[:, 0]
        start_index = (x_column >= x_value).idxmax()
        end_index = start_index
        T=self.time_window

        for i in range(start_index + 1, len(df)):
            if x_column[i] - x_value >= T:
                break
            end_index = i
        if start_index+(T*100) > len(df):
            start_index=len(df)-(T*100)
            end_index=len(df)

        filtered_df = df.iloc[start_index:start_index + (T*100)]
        return filtered_df

    def plot_csv_data(self, df):
        """
        CSV 파일 데이터를 그래프로 시각화하는 함수.
        """
        x_values = df.iloc[:, 0]
        y_values = df.iloc[:, 1:]

        plt.figure(figsize=(10, 6))
        lines = []

        for column in y_values.columns:
            line, = plt.plot(x_values, y_values[column], label=column)
            lines.append(line)

        plt.title(f'{self.Y_label[int(self.count/self.data_set_per_label)]}-{self.count % self.data_set_per_label + 1}th CSV Data Visualization')
        plt.xlabel(df.columns[0])
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)

        self.count=self.count+1
        mplcursors.cursor(lines, hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(f'시간: {sel.target[0]:.2f}')
        )

        def on_click(event):
            if event.inaxes:
                clicked_x = event.xdata
                print(f"Clicked X Value: {clicked_x:.2f}")

                # 기존 filtered_df를 processed_array로 변환
                filtered_df = self.filter_data_by_time(df, clicked_x)
                #print("Filtered Data:")
                #print(filtered_df)

            # NumPy 배열로 변환
                processed_array = self.make_csv_array(filtered_df)
                #processed_array=processed_array[:,1:]
                print("Processed Array:")
                print(processed_array)

            # 새로운 데이터를 raw_array에 추가
                self.raw_array.append(processed_array)
            
            # 최종 3차원 배열 갱신
                plt.close()

        plt.gcf().canvas.mpl_connect("button_press_event", on_click)
        plt.show()  

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