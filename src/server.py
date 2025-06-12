from flask import Flask, request, jsonify, session, render_template, Response
from flask_session import Session
from flask_cors import CORS
import makenumpyfile, train_model, test_model
import numpy as np
import uuid
from datetime import timedelta
import os
import joblib
from queue import Queue
from threading import Thread
import time

app = Flask(__name__)

# 세션 설정
app.config['SECRET_KEY'] = 'your_secret_key'  # 세션 암호화를 위한 키
app.config['SESSION_TYPE'] = 'filesystem'    # 세션 데이터를 파일에 저장
Session(app)
CORS(app)

app.permanent_session_lifetime = timedelta(days=1)

@app.route('/')
def index():
    return render_template('client.html')

@app.route('/initialize', methods=['GET'])
def initialize():
    # 세션에 client_id가 없으면 새로 생성
    if 'client_id' not in session:
        session['client_id'] = str(uuid.uuid4())  # 고유한 UUID 생성
    return jsonify({"client_id": session['client_id']})


@app.route('/submit-labels', methods=['POST'])
def submit_labels():
    client_id = session.get('client_id')
    if 'client_id' not in session:
        session['client_id'] = str(uuid.uuid4())  # 고유한 UUID 생성

    data = request.get_json()
    labels = data.get('labels', [])

    session['labels'] = labels
    
    print(f"클라이언트 {client_id}의 데이터: {labels}")
    return jsonify({"message": "Labels 저장 완료!"})
    

@app.route("/input_raw_data", methods=["POST"])
def make_data_from_csv():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid data format. Expected a dictionary."}), 400


    session["data_set"], session["Y_label"]=makenumpyfile.make_data_csv(data["folder_path"], data["file_name"], data_set_per_label=10, time_window=3)

    return jsonify({
        "message": "Data saved successfully",
        "Y_label": session["Y_label"]  # 현재 세션 데이터 반환
    })


@app.route("/input_npy_data", methods=["POST"])
def make_data_from_npy():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "파일이 업로드되지 않았습니다."})
    file = request.files['file']

    # 파일 확장자 확인
    if not file.filename.endswith('.npy'):
        return jsonify({"success": False, "message": "NPY 파일만 업로드 가능합니다."})
    
    data = np.load(file)

    # 3차원 배열인지 확인
    if len(data.shape) < 3:
        return jsonify({"success": False, "message": "파일이 3차원 배열이 아닙니다."})

    total_count = data.shape[0]
    session["data_set"]=data

    return jsonify({
        "success": True,
        "message": "Data saved successfully",
        "Y_label": session["labels"],  # 현재 세션 데이터 반환
        "total_count": total_count
    })

@app.route("/set_train", methods=["POST"])
def set_train():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
    data=request.json
    session["stat_var"]=data.get('stat_var')
    session["fft_var"]=data.get('fft_var')

    return jsonify({'message': 'Data saved successfully!'})

@app.route("/train_data", methods=["GET"])
def train_data():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
    t_data_set = session["data_set"]
    t_labels= session["labels"]
    stat_var=session["stat_var"]
    fft_var=session["fft_var"]


    def generate():
        q = Queue()
        # 콜백 함수 정의
        def progress_callback(message):
            q.put(message)

        def run_training():
            model, label_encoder = train_model.train_m(t_data_set, t_labels, stat_variable=stat_var, fft_variable=fft_var, callback=progress_callback)
            # 모델 및 라벨 인코더 저장
            
            os.makedirs('tmp', exist_ok=True)
            client_dir = os.path.join("tmp", client_id)
            os.makedirs(client_dir, exist_ok=True)

            model_path = os.path.join(client_dir, "model.pkl")
            label_path = os.path.join(client_dir, "label_encoder.pkl")
            joblib.dump(model, model_path)
            joblib.dump(label_encoder, label_path)
            
            q.put(None)  # 완료 신호

        Thread(target=run_training).start()

        while True:
            message = q.get()
            if message is None:
                break
            yield f"data: {message}\n\n"

        yield "data: Training completed.\n\n"

    return Response(generate(), content_type="text/event-stream")


@app.route("/input_npy_data_test", methods=["POST"])
def make_data_from_npy_test():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "파일이 업로드되지 않았습니다."})
    file = request.files['file']

    # 파일 확장자 확인
    if not file.filename.endswith('.npy'):
        return jsonify({"success": False, "message": "NPY 파일만 업로드 가능합니다."})
    
    data = np.load(file)

    # 3차원 배열인지 확인
    if len(data.shape) < 3:
        return jsonify({"success": False, "message": "파일이 3차원 배열이 아닙니다."})

    total_count = data.shape[0]
    session["test_set"]=data

    return jsonify({
        "success": True,
        "message": "Data saved successfully",
        "Y_label": session["labels"],  # 현재 세션 데이터 반환
        "total_count": total_count
    })


@app.route("/test", methods=["GET"])
def test():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
       
    datatest_list=session["test_set"]
    y_label=session["labels"]
    stat_var=session["stat_var"]
    fft_var=session["fft_var"]    

    client_dir = os.path.join("tmp", client_id)
    model_path = os.path.join(client_dir, "model.pkl")
    label_path = os.path.join(client_dir, "label_encoder.pkl")
    
    if os.path.exists(model_path) and os.path.exists(label_path):
        model = joblib.load(model_path)
        label_encoder = joblib.load(label_path)
        predicted_class=test_model.test_m(datatest_list, model, label_encoder, y_label, stat_variable=stat_var, fft_variable=fft_var)
        def generate():
            for i, pred in enumerate(predicted_class):
                yield f"data: Test Sample {i+1}: Predicted Motion = {label_encoder.inverse_transform([pred.item()])}\n\n"
                time.sleep(1)  # 1초 대기 (연속적인 메시지 전송 시뮬레이션)
            yield "data: 총 결과는 이렇답니다~\n\n"  # 마지막 메시지

        return Response(generate(), content_type="text/event-stream")
        
    else:
        raise FileNotFoundError("Model or Label Encoder not found!")


@app.route('/clear', methods=['POST'])
def clear_session():
    # 현재 클라이언트의 세션 초기화
    session.clear()
    return jsonify({"message": "Session cleared!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


@app.route('/login', methods=['POST'])
def login():
    # 클라이언트가 로그인하면 세션 ID 생성
    client_id = request.json.get('client_id')
    if not client_id:
        return jsonify({"error": "client_id is required"}), 400

    session['client_id'] = client_id
    return jsonify({"message": "Session initialized", "client_id": client_id}), 200




if __name__ == '__main__':
    # SSL 인증서와 키 파일 경로 설정
    app.run(ssl_context=('C:/Users/전재형/ca/server.crt',
                         'C:/Users/전재형/ca/server.key'),
            host='0.0.0.0', port=443)