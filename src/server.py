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
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# 세션 설정
app.config['SECRET_KEY'] = 'your_secret_key'  # 세션 암호화를 위한 키
app.config['SESSION_TYPE'] = 'filesystem'    # 세션 데이터를 파일에 저장
app.config['SESSION_COOKIE_SECURE'] = True # HTTPS 환경에서만 쿠키 전송
app.config['SESSION_COOKIE_HTTPONLY'] = True # HTTP 요청에서만 쿠키 접근 가능
app.config['SESSION_COOKIE_SAMESITE'] = 'None' # SameSite 속성 설정

Session(app)
CORS(app, supports_credentials=True, origins=['http://localhost:3000']) 
app.permanent_session_lifetime = timedelta(days=1)

PARAM_COUNTS = {
    "GRU": 4,
    "RNN": 4,
    "KNN": 2,
    "SVM": 2
}

@app.route('/')
def index():
    return render_template('/client.html')

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
    
    # tmp/{client_id} 폴더 생성
    client_tmp_dir = os.path.join('tmp', client_id)
    os.makedirs(client_tmp_dir, exist_ok=True)

    # 파일 업로드 처리
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist('files')
    
    saved_files = []
    for file in files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(client_tmp_dir, filename)
        file.save(save_path)
        saved_files.append(filename)

    # 디버깅: 세션에서 라벨 확인
    labels = session.get('labels')
    print(f"[DEBUG] 전달할 labels: {labels}")  # 
    print(f"[DEBUG] client_id: {client_id}, session['labels']: {labels}")

    if not labels:
        print("[ERROR] 세션에 라벨이 없습니다!")
        return jsonify({"error": "Labels not found in session. 먼저 라벨을 제출하세요."}), 400
    
    # 파일명에서 공통 prefix 추출 (예: RawData)
    if not saved_files:
        return jsonify({"error": "No files uploaded"}), 400
    base_name = os.path.splitext(saved_files[0])[0]

    labels = session.get('labels')
    num_labels = len(labels)
    files_per_label = 10  # 라벨당 10개로 고정

    if len(saved_files) != num_labels * files_per_label:
        return jsonify({"error": f"파일 개수는 라벨 수({num_labels}) x 10 = {num_labels*10}개여야 합니다."}), 400

    # makenumpyfile.make_data_csv 호출
    try:
        data_set, y_label = makenumpyfile.make_data_csv(
            folder_path=client_tmp_dir,
            file_name=base_name,
            data_set_per_label=files_per_label,
            time_window=3,
            labels=labels
        )
        session["data_set"] = data_set
        session["Y_label"] = y_label
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
    "message": "Data saved successfully",
    "Y_label": y_label.tolist() if hasattr(y_label, "tolist") else y_label
}
)

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

@app.route("/select_model", methods=["POST"])
def select_model():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
    data=request.json
    selected_model = data.get('model')
    if selected_model:
        print(f"선택된 모델: {selected_model}")
        if selected_model not in PARAM_COUNTS:
            return jsonify(ok=False, error="존재하지 않는 모델"), 400
        session["model"] = selected_model
        session.pop("params", None)
        return jsonify({'message': f'{selected_model} 모델이 저장되었습니다!'})
    return jsonify({'message': '모델 선택에 실패했습니다.'}), 400

@app.route("/set_params", methods=["POST"])
def set_params():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
    if "model" not in session:
        return jsonify(ok=False, error="모델을 먼저 선택하세요"), 400

    model   = session["model"]
    needed  = PARAM_COUNTS[model]         # 필요한 개수
    params  = request.json.get("params", [])

    if len(params) < needed:
        return jsonify(ok=False,
                       error=f"{needed}개의 값이 필요합니다"), 400

    # 파라미터 타입 변환 (정수/실수로)
    if model in ["GRU", "RNN"]:
        # [test_size, batch_size, learning_rate, num_epochs]
        params = [
            float(params[0]),   # test_size
            int(params[1]),     # batch_size
            float(params[2]),   # learning_rate
            int(params[3])      # num_epochs
        ]
    elif model in ["KNN", "SVM"]:
        # [test_size, n_neighbors]
        params = [
            float(params[0]),   # test_size
            int(params[1])      # n_neighbors
        ]

    session["params"] = params[:needed]
    return jsonify({'message': '매개변수 설정 완료!.'})

    
@app.route("/train_data", methods=["GET"])
def train_data():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
    t_data_set = session["data_set"]
    t_labels= session["labels"]
    stat_var=session["stat_var"]
    fft_var=session["fft_var"]
    selected_model=session["model"]
    params=session["params"]

    def generate():
        q = Queue()
        # 콜백 함수 정의
        def progress_callback(message):
            q.put(message)

        def run_training():
            model, label_encoder = train_model.train_NN(
                selected_model, t_data_set, t_labels,
                stat_variable=stat_var, fft_variable=fft_var, 
                _test_size=params[0], _batch_size=params[1], _learning_rate=params[2], _num_epochs=params[3],  # 수정: [3] → params[3]
                callback=progress_callback
            )
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

    def generate_M():
        q = Queue()
        # 콜백 함수 정의
        def progress_callback(message):
            q.put(message)

        def run_training():
            model, label_encoder = train_model.train_m(selected_model, t_data_set, t_labels, stat_variable=stat_var, fft_variable=fft_var,
                                                       _test_size=params[0], _n_neighbors=params[1], callback=progress_callback)
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

    if selected_model == 'KNN' or selected_model == 'SVM':
        return Response(generate_M(), content_type="text/event-stream")
    else:
        return Response(generate(), content_type="text/event-stream")


@app.route("/input_npy_data_test", methods=["POST"]) #테스트 할 데이터를 넘파이로 받아줌. input _ csv requst 만들어야됨. 
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
    selected_model=session["model"]

    client_dir = os.path.join("tmp", client_id)
    model_path = os.path.join(client_dir, "model.pkl")
    label_path = os.path.join(client_dir, "label_encoder.pkl")
    
    if os.path.exists(model_path) and os.path.exists(label_path):
        model = joblib.load(model_path)
        label_encoder = joblib.load(label_path)
        if selected_model == 'SVM' or selected_model == 'KNN':
            predicted_class=test_model.test_m(datatest_list, model, label_encoder, y_label, stat_variable=stat_var, fft_variable=fft_var)
        else:
            predicted_class=test_model.test_NN(datatest_list, model, label_encoder, y_label, stat_variable=stat_var, fft_variable=fft_var)

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
# 사용자 별로 쌓인 세션을 관리하기 위해 Flask-Session 

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
    app.run(ssl_context=('/Users/songjunha/certificate.crt',
                         '/Users/songjunha/private.key'),
            host='0.0.0.0', port=443)