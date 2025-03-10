from flask import Flask, request, jsonify, session
from flask_session import Session
from ..src import makenumpyfile, train_model, test_model

app = Flask(__name__)

# 세션 설정
app.config['SECRET_KEY'] = 'your_secret_key'  # 세션 암호화를 위한 키
app.config['SESSION_TYPE'] = 'filesystem'    # 세션 데이터를 파일에 저장
Session(app)


@app.route("/input_raw_data", methods=["POST"])
def make_data():
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
def make_data():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid data format. Expected a dictionary."}), 400

    session["data_set"]=data["data_set"]
    session["Y_label"]=data["Y_label"]

    return jsonify({
        "message": "Data saved successfully",
        "Y_label": session["Y_label"]  # 현재 세션 데이터 반환
    })


@app.route("/train_data", methods=["POST"])
def train_data():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    train_model.train_m(session["data_set"], session["Y_label"])
    return jsonify({
        "message": "Data trained successfully"
    })


@app.route("/test", methods=["POST"])
def test():
    client_id = session.get('client_id')
    if not client_id:
        return jsonify({"error": "Session not initialized"}), 401
    
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid data format. Expected a dictionary."}), 400
    
    test_model.test_m(data["test"], session["Y_label"])
    return jsonify({
        "message": "Data testes successfully"
    })


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
