document.getElementById("start-learning").addEventListener("click", function() {
    // 서버에 요청 보내기
    fetch("http://127.0.0.1:5000/initialize")
        .then(response => response.json()) // 응답을 JSON으로 변환
        .then(data => {
            if (data.client_id) {
                // 클라이언트 ID를 화면에 표시
                document.getElementById("client-id").textContent = data.client_id;
            } else {
                alert("클라이언트 ID를 가져오는 데 실패했습니다.");
            }
        })
        .catch(error => {
            console.error("오류 발생:", error);
            alert("서버와 통신 중 오류가 발생했습니다.");
        });
});

document.addEventListener("DOMContentLoaded", () => {
    const labelInput = document.getElementById("label-input");
    const addLabelBtn = document.getElementById("add-label-btn");
    const labelContainer = document.getElementById("label-container");
    const submitBtn = document.getElementById("submit-btn");
    const labelSummary = document.getElementById("label-summary");
  
    let labels = []; // 입력된 label들을 저장하는 배열
  
    // Label 추가 버튼 클릭 이벤트
    addLabelBtn.addEventListener("click", () => {
      const label = labelInput.value.trim(); // 공백 제거
      if (label) {
        labels.push(label); // 배열에 추가
        renderLabels(); // Label 박스 다시 렌더링
        labelInput.value = ""; // 입력창 비우기
      }
    });
  
    // Label 박스 렌더링 함수
    function renderLabels() {
      labelContainer.innerHTML = ""; // 기존 내용을 초기화
      labels.forEach((label, index) => {
        const labelBox = document.createElement("div");
        labelBox.className = "label-box";
        labelBox.textContent = label;
  
        // X 버튼 추가
        const removeBtn = document.createElement("button");
        removeBtn.className = "remove-btn";
        removeBtn.textContent = "x";
        removeBtn.addEventListener("click", () => {
          labels.splice(index, 1); // 배열에서 해당 label 제거
          renderLabels(); // 다시 렌더링
        });
  
        labelBox.appendChild(removeBtn); // Label 박스에 X 버튼 추가
        labelContainer.appendChild(labelBox); // Label 박스를 컨테이너에 추가
      });
    }
  
    // 확인 버튼 클릭 이벤트
    submitBtn.addEventListener("click", () => {
      if (labels.length > 0) {
        // 서버에 Label 데이터 전송
        labelSummary.textContent = `총 입력된 Label: ${labels.join(", ")}`;
        fetch("http://127.0.0.1:5000/submit-labels", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ labels }), // labels 배열을 JSON으로 변환하여 전송
        })
          .then((response) => response.json())
          .then((data) => {
            alert("서버에 저장되었습니다: " + data.message);
            labels = []; // 배열 초기화
            renderLabels(); // 화면 초기화
          })
          .catch((error) => {
            console.error("오류 발생:", error);
          });
      } else {
        alert("Label을 입력하세요!");
      }
    });
  });

const csvButton = document.getElementById("csvButton");
const npyButton = document.getElementById("npyButton");
const fileInput = document.getElementById("fileInput");
const result = document.getElementById("result");

csvButton.addEventListener("click", () => {
    alert("CSV 파일 업로드는 아직 구현되지 않았습니다.");
});

// NPY 버튼 클릭 이벤트
npyButton.addEventListener("click", () => {
    fileInput.accept = ".npy"; // NPY 파일만 선택 가능
    fileInput.click(); // 파일 선택 창 열기
});

// 파일 선택 이벤트
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;

    // FormData를 이용해 파일 전송
    const formData = new FormData();
    formData.append("file", file);

    // 서버로 파일 업로드
    fetch("/input_npy_data", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            result.textContent = `총 데이터 개수: ${data.total_count}`;
        } else {
            result.textContent = `오류: ${data.message}`;
        }
    })
    .catch(err => {
        result.textContent = "파일 업로드 중 오류가 발생했습니다.";
        console.error(err);
    });
});

document.getElementById('select_preprocess').addEventListener('click', () => {
  // 2진법 선택 데이터 계산
  let binarySelection = 0;
  for (let i = 0; i <= 6; i++) {
      const checkbox = document.getElementById(`option${i}`);
      if (checkbox.checked) {
          binarySelection |= (1 << (i));  // 해당 비트 설정
      }
  }

  // FFT 선택 여부
  const fftSelected = document.getElementById('fft').checked;

  // 서버로 데이터 전송
  fetch('/set_train', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({
          stat_var: binarySelection,
          fft_var: fftSelected
      }),
  })
  .then(response => response.json())
  .then(data => {
      alert(data.message);
  })
  .catch(error => {
      console.error('Error:', error);
  });
});

document.getElementById('submit-button').addEventListener('click', function () {
    const selectedModel = document.querySelector('input[name="model"]:checked');

    if (!selectedModel) {
        alert('모델을 선택해주세요!');
        return;
    }

    const modelName = selectedModel.value;

    // SVM과 KNN 구분 메시지
    if (modelName === 'SVM' || modelName === 'KNN') {
        console.log(`${modelName}은(는) 뉴럴 네트워크가 아닙니다.`);
    } else {
        console.log(`${modelName}은(는) 뉴럴 네트워크 기반 모델입니다.`);
    }

    // 서버로 데이터 전송
    fetch('/select_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model: modelName }),
    })
        .then((response) => response.json())
        .then((data) => {
            document.getElementById('response-message').innerText = data.message;
        })
        .catch((error) => {
            console.error('Error:', error);
        });
});


/* ---------- 2) 파라미터 저장 ---------- */
document.getElementById("set_params").addEventListener("click", async () => {
    // 입력된 매개변수들을 배열로 수집
    const inputs = document.querySelectorAll(".param_input");
    const params = Array.from(inputs).map(input => parseFloat(input.value.trim()));

    // 빈 값이 있을 경우 경고
    if (params.some(param => param === "")) {
        alert("Please fill in all parameter fields.");
        return;
    }

    // 서버로 매개변수 전송
    const response = await fetch('/set_params', {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params }),
    });

    const result = await response.json();
    document.getElementById("params_status").textContent = result.message || result.error;
});


// 학습 시작
document.getElementById('startTraining').addEventListener('click', () => {
  /*const epochs = document.getElementById('epochs').value || 10;
  const batchSize = document.getElementById('batchSize').value || 32;*/

  const eventSource = new EventSource(`/train_data`);
  const output = document.getElementById('output');

  eventSource.onmessage = (event) => {
      output.textContent += event.data + '\n';
  };

  eventSource.onerror = () => {
      console.error("연결 오류 발생");
      eventSource.close();
  };
});

const csvButtontest = document.getElementById("csvButtontest");
const npyButtontest = document.getElementById("npyButtontest");
const fileInputtest = document.getElementById("fileInputtest");
const resulttest = document.getElementById("resulttest");

csvButtontest.addEventListener("click", () => {
    alert("CSV 파일 업로드는 아직 구현되지 않았습니다.");
});

// NPY 버튼 클릭 이벤트
npyButtontest.addEventListener("click", () => {
    fileInputtest.accept = ".npy"; // NPY 파일만 선택 가능
    fileInputtest.click(); // 파일 선택 창 열기
});

// 파일 선택 이벤트
fileInputtest.addEventListener("change", () => {
    const file = fileInputtest.files[0];
    if (!file) return;

    // FormData를 이용해 파일 전송
    const formData = new FormData();
    formData.append("file", file);

    // 서버로 파일 업로드
    fetch("/input_npy_data_test", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            resulttest.textContent = `총 데이터 개수: ${data.total_count}`;
        } else {
            resulttest.textContent = `오류: ${data.message}`;
        }
    })
    .catch(err => {
        resulttest.textContent = "파일 업로드 중 오류가 발생했습니다.";
        console.error(err);
    });
});

document.getElementById('startTesting').addEventListener('click', () => {
  /*const epochs = document.getElementById('epochs').value || 10;
  const batchSize = document.getElementById('batchSize').value || 32;*/

  const eventSource = new EventSource(`/test`);
  const output = document.getElementById('testresult');

  eventSource.onmessage = (event) => {
      output.textContent += event.data + '\n';
  };

  eventSource.onerror = () => {
      console.error("연결 오류 발생");
      eventSource.close();
  };
});