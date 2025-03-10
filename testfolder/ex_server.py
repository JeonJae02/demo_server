from flask import Flask, request, jsonify

app = Flask(__name__)

coefficients = []  # 클라이언트가 보낸 숫자들을 저장 (다항식 계수)

@app.route("/add_number", methods=["POST"])
def add_number():
    """클라이언트가 숫자를 보내면 서버에 저장"""
    data = request.json
    number = data.get("number")

    if number is None:
        return jsonify({"error": "Missing 'number' parameter"}), 400

    coefficients.append(number)  # 리스트에 숫자 추가
    return jsonify({"message": f"숫자 {number} 추가됨", "coefficients": coefficients})


@app.route("/calculate", methods=["POST"])
def calculate():
    """최종 입력값을 받아서 다항식 계산"""
    if len(coefficients) < 3:  # 최소한 3개의 숫자가 있어야 ax² + bx + c 형태 가능
        return jsonify({"error": "최소 3개의 숫자가 필요합니다."}), 400

    data = request.json
    x = data.get("x")

    if x is None:
        return jsonify({"error": "Missing 'x' parameter"}), 400

    # 다항식 예제: ax² + bx + c (3개의 계수를 사용)
    a, b, c = coefficients[-3:]  # 최근 3개의 숫자로 다항식 구성
    result = a * (x ** 2) + b * x + c  # 계산

    return jsonify({"function": f"{a}x² + {b}x + {c}", "result": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
