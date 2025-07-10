from RawPreProcessing import rawpreprocessing
import numpy as np
import os
import pandas as pd

# 클래스 인스턴스 생성
def make_data_csv(folder_path, file_name, data_set_per_label=10, time_window=3, labels=None):
    print(f"[DEBUG] make_data_csv로 전달된 labels: {labels}")  # 추가
    base_name, _ = os.path.splitext(file_name)
    processor = rawpreprocessing(
        data_set_per_label=data_set_per_label, 
        time_window=time_window ,
          labels=labels)  # labels를 전달 label당 10개 , 창의 크기 3
    num_data_set=processor.num_data_set
    # 데이터 처리 및 시각화 반복
    for i in range(1, num_data_set + 1):
        file_path=os.path.join(folder_path, f"RawData{i}.csv")
        if not os.path.exists(file_path):
            print(f"파일이 존재하지 않습니다: {file_path}")
            continue
        #result_df = processor.remove_edges_from_csv(file_path)
        #file_path = f"C:/Users/user/OneDrive/바탕 화면/test/rawdataset/RawData{i}.csv"
        # CSV 파일에서 양 끝 10% 데이터 제거
        #result_df = processor.remove_edges_from_csv(file_path)

        # 데이터 시각화 & T초로 자르기
        #processor.plot_csv_data(result_df)

        # 새로운 CSV 파일로 저장
        #trimmed_file_path = f"C:/Users/교육생-PC08/trimmed_data/trimmed_data{i}.csv"
        #result_df.to_csv(trimmed_file_path, index=False)
        df = pd.read_csv(file_path)
        processed_array = processor.make_csv_array(df)
        if processed_array is not None:
            processor.raw_array.append(processed_array)
    # 최종 3차원 배열 생성
    print(f"최종 3차원 배열 생성 중...")
    total_array=processor.make_total_array()
    y_label=processor.Y_label
    np.save('train_data2.npy', total_array)
    return total_array, y_label

#data_set, Y_label=make_data_csv(folder_path="C:/Users/user/OneDrive/바탕 화면/test/rawdataset", file_name="RawData")

def upload_and_process_files(session, files):
    client_tmp_dir = "C:/Users/user/AppData/Local/Temp"
    os.makedirs(client_tmp_dir, exist_ok=True)

    saved_files = []
    for file in files:
        file.save(os.path.join(client_tmp_dir, file.filename))
        saved_files.append(file.filename)

    labels = session.get('labels')
    num_labels = len(labels)
    files_per_label = 10  # 라벨당 10개로 고정

    # 파일 개수 체크
    if len(saved_files) != num_labels * files_per_label:
        return jsonify({"error": f"파일 개수는 라벨 수({num_labels}) x 10 = {num_labels*10}개여야 합니다."}), 400

    if not saved_files:
        return jsonify({"error": "No files uploaded"}), 400
    base_name = os.path.splitext(saved_files[0])[0]

    try:
        data_set, y_label = make_data_csv(
            folder_path=client_tmp_dir,
            file_name=base_name,
            data_set_per_label=files_per_label,  # 항상 10개로 고정
            time_window=3,
            labels=labels
        )
        # 필요하다면 반환값 처리
    except Exception as e:
        print(f"[ERROR] make_data_csv 실행 중 오류: {e}")
        return