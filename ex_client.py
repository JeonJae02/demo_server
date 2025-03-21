import requests

server_url = "http://127.0.0.1:5000"

# 1️⃣ 여러 개의 숫자 입력 (계수 저장)
print("서버에 보낼 숫자를 입력하세요. (최소 3개)")
coefficients = []

while True:
    num_input = input("숫자 입력 (완료하려면 'done' 입력): ")
    if num_input.lower() == "done":
        break
    try:
        number = float(num_input)
        coefficients.append(number)
        response = requests.post(f"{server_url}/add_number", json={"number": number})
        print("서버 응답:", response.json())
    except ValueError:
        print("⚠ 숫자를 입력하세요!")

if len(coefficients) < 3:
    print("❌ 최소 3개의 숫자가 필요합니다. 프로그램 종료.")
    exit()

# 2️⃣ 최종 입력 값 (x) 입력
while True:
    x_input = input("\n최종 입력 값 (x)을 입력하세요: ")
    try:
        x_value = float(x_input)
        break
    except ValueError:
        print("⚠ 숫자를 입력하세요!")

# 3️⃣ 최종 계산 요청
response = requests.post(f"{server_url}/calculate", json={"x": x_value})
print("\n📌 최종 계산 결과:", response.json())
