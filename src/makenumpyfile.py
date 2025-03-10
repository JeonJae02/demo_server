from RawPreProcessing import rawpreprocessing
import numpy as np
import os
# 클래스 인스턴스 생성
def make_data_csv(folder_path, file_name, data_set_per_label=10, time_window=3):
    processor = rawpreprocessing(data_set_per_label=data_set_per_label, time_window=time_window) # label당 10개 , 창의 크기 3
    num_data_set=processor.num_data_set
    # 데이터 처리 및 시각화 반복
    for i in range(1, num_data_set + 1):
        file_path=os.path.join(folder_path, f"{file_name}{i}.csv")
        #file_path = f"C:/Users/user/OneDrive/바탕 화면/test/rawdataset/RawData{i}.csv"
    
        # CSV 파일에서 양 끝 10% 데이터 제거
        result_df = processor.remove_edges_from_csv(file_path)

        # 데이터 시각화 & T초로 자르기
        processor.plot_csv_data(result_df)

        # 새로운 CSV 파일로 저장
        #trimmed_file_path = f"C:/Users/교육생-PC08/trimmed_data/trimmed_data{i}.csv"
        #result_df.to_csv(trimmed_file_path, index=False)

    # 최종 3차원 배열 생성
    total_array=processor.make_total_array()
    y_label=processor.Y_label
    np.save('train_data1.npy', total_array)
    return total_array, y_label

#data_set, Y_label=make_data_csv(folder_path="C:/Users/user/OneDrive/바탕 화면/test/rawdataset", file_name="RawData")