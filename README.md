# 📚 snnTorch-Tutorials: 개인 학습 저장소

이 저장소는 PyTorch 기반의 스파이킹 신경망(Spiking Neural Networks, SNN) 라이브러리인 `snnTorch`를 학습하며 정리한 개인 공부용 저장소입니다. 공식 튜토리얼을 따라가며 코드 예시를 실행하고, 주요 개념을 이해하며 주석과 설명을 추가했습니다.

## 🌟 프로젝트 목표

* `snnTorch` 라이브러리의 핵심 기능 및 사용법 숙달
* 스파이킹 신경망(SNN)의 기본 개념 (스파이크, 래스터 플롯, 발화율/지연 시간/델타 변조 부호화 등) 이해
* PyTorch 환경에서 SNN 모델 구축 및 훈련 과정 익히기
* 향후 SNN 관련 연구 및 프로젝트를 위한 기초 다지기

## 📁 저장소 구성

이 저장소는 `snnTorch` 공식 튜토리얼의 각 섹션에 맞춰 구성될 예정입니다.

* `Tutorial-1-Spike-Encoding/`: 스파이크 생성 및 시각화 (spikegen, spikeplot) 관련 튜토리얼
* `Tutorial-2-The-Leaky-Intergrate-and-Fire-Neuron/`: 스파이크 생성 및 시각화 (spikegen, spikeplot) 관련 튜토리얼  
* `Tutorial-3-A-Feedforward-Spiking-Neural-Network/`: 스파이크 생성 및 시각화 (spikegen, spikeplot) 관련 튜토리얼
* `Tutorial-4-2nd-Roder-Spiking-Neuron-Models/`: 스파이크 생성 및 시각화 (spikegen, spikeplot) 관련 튜토리얼
* `Tutorial-5-Training-Spiking-Neural-Network-with-snnTourch/`: 스파이크 생성 및 시각화 (spikegen, spikeplot) 관련 튜토리얼
* `Tutorial-6-Surrogate-Gradient-Descent-in-a-Convolutional-SNN/`: 스파이크 생성 및 시각화 (spikegen, spikeplot) 관련 튜토리얼
* `Tutorial-7-Neuromorphic-Datasets-with-Tonic+snnTourch/`: 스파이크 생성 및 시각화 (spikegen, spikeplot) 관련 튜토리얼
* `Exoplanet-Hunter:Finding-Planets-Using-Light-Intensity/`: 스파이크 생성 및 시각화 (spikegen, spikeplot) 관련 튜토리얼
* `The-Foward-Forward-Algorithm-with-a-Spiking-Neural-Network/`: 스파이크 생성 및 시각화 (spikegen, spikeplot) 관련 튜토리얼
* `Accelerating-snnTourch-On-IPUs/`: 스파이크 생성 및 시각화 (spikegen, spikeplot) 관련 튜토리얼
각 폴더 안에는 해당 튜토리얼의 파이썬 코드(`.py` 파일 또는 `.ipynb` Jupyter Notebook 파일)와 함께 필요한 경우 추가적인 설명이나 결과 이미지가 포함될 수 있습니다.

## 🛠️ 환경 설정

이 프로젝트를 실행하기 위해서는 다음과 같은 환경이 필요합니다.

1.  **Python 3.x**
2.  **PyTorch**
3.  **snnTorch**
4.  **Matplotlib**
5.  **FFmpeg (애니메이션 저장을 위해 필요)**

### 설치 방법

```bash
# 가상 환경 생성 (권장)
python -m venv snntorch_env
source snntorch_env/bin/activate  # Linux/macOS
# snntorch_env\Scripts\activate # Windows

# 필요한 라이브러리 설치
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) # 또는 GPU 환경에 맞게
pip install snntorch matplotlib
pip install jupyter  # Jupyter Notebook 사용 시

# FFmpeg 설치 (Windows 예시, 다른 OS는 해당 설치 가이드 참고)
# 1. [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) 에서 Windows 빌드 (예: gyan.dev의 "full_build") 다운로드
# 2. 다운로드한 ZIP 파일 압축 해제 (예: C:\ffmpeg)
# 3. 코드 내에서 ffmpeg.exe 경로 설정 (예: plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe')
