# 1.1. Import packages and setup environment
import snntorch as snn
import torch
from torchvision import datasets, transforms
# torchvision : PyTourch에서 이미지 및 비디오 처리를 위해 특별히 설계된 라이브러리, 이미지 관련 데이터셋, 모델, 변환(Transform) 기능을 제공
    # datasets : 다양한 표준 데이터셋(MNIST, CIFAR10등)을 쉽게 다운로드하고, 로드하는 기능 제공
    # transforms : 이미지 데이터를 신경망 모델이 학습하기에 적합한 형태로 변환(전처리)하는 다양한 함수 제공
from snntorch import utils
# snntorch.utils 모듈은 스파이킹 신경망을 다룰 때 유용하게 사용할 수 있는 다양한 유틸리티 함수들을 제공합니다.
# 데이터셋을 다루거나 특정 계산을 수행할 때 편리한 기능들이 포함되어 있다.

from torch.utils.data import DataLoader

# torch.utils.data: PyTorch에서 데이터 로딩을 위한 유틸리티를 제공하는 핵심 모듈입니다.
# DataLoader: 데이터셋(Dataset 객체)을 모델이 학습할 수 있는 형태로 효율적으로 로드해주는 역할을 합니다.
# 특히, 데이터를 배치(batch) 단위로 묶고, 데이터를 섞거나(shuffle), 필요하다면 여러 워커(worker) 프로세스를 사용하여 데이터 로딩 속도를 높이는 기능
# 등을 제공합니다.


# Training Parameters
batch_size=128
data_path='/tmp/data/mnist'
num_classes = 10  # MNIST has 10 output classes

# Torch Variables
dtype = torch.float

# Define a transform
"""

transforms.Compose(self, transform): 여러개의 변환을 순서대로 연결하여 한번에 적용할 수 있는 유용한 기능
transforms.Resize(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True) : 이미지 RESIZE 전처리 함수 
- size : (필수 파라미터) 이미지의 출력 크기 정의 (H,W)
- interpolation : 보간 방법 지정 ex ) InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.BICUBIC, InterpolationMode.LANCZOS
- max_size : size 파라미터가 정수로 주어졌을 때만 적용됩니다. 이미지의 긴 변이 이 max_size 값을 초과하지 않도록 제한
- antialias : 앤티앨리어싱(Anti-aliasing)을 적용할지 여부를 결정

transforms.Grayscale(self, num_output_channels=1) :  컬러 이미지를 회색조(Grayscale) 이미지로 변환
- num_output_channels : 출력될 회색조 이미지의 채널 수를 지정
transforms.ToTensor(): PIL(Python Imaging Library) 이미지나 NumPy ndarray를 PyTorch FloatTensor로 변환하고, 픽셀 값의 범위를 자동으로 정규화합니다
transforms.Normalize(self, mean, std, inplace=False):  주어진 평균(mean)과 표준편차(std)를 사용하여 PyTorch 텐서 이미지를 채널별로 정규화
-mean : 각 채널에 대해 뺄 평균 값의 시퀀스(리스트 또는 튜플)입니다.
-std : 각 채널에 대해 나눌 표준편차 값의 시퀀스
-intplacse : 연산을 수행한 결과를 원본 텐서에 직접 덮어쓸지(True), 아니면 새로운 텐서로 반환할지(False)를 결정
"""

# 1.2 Download Dataset
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])


"""
datasets.MNIST는 torchvision.datasets 모듈 안에 있는 클래스로, MNIST 데이터셋을 손쉽게 다룰 수 있도록 미리 만들어진 도구
data_path : NIST 데이터셋 파일이 저장될 위치나 이미 저장되어 있는 위치를 지정
train : 값을 True로 설정하면 MNIST 학습(training) 데이터셋을 불러오겠다는 의미 train=True로 학습 데이터를 사용 모델의 성능을 평가할 때는 train=False로 테스트 데이터를 사용
download : 이 값을 True로 설정하면, 지정된 data_path에 MNIST 데이터셋이 없을 경우 자동으로 웹에서 데이터셋을 다운로드
transform : 이전에 우리가 transforms.Compose를 사용하여 정의했던 **일련의 이미지 전처리 파이프라인(transform 변수)**이 들어가는 자리


"""
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)


subset = 10
mnist_train = utils.data_subset(mnist_train, subset)
# mnist_train
# 1.3 Create DataLoaders


"""
 DataLoader 객체를 생성하고, 이를 train_loader라는 변수에 할당하는 코드입니다. 
 이제 이 train_loader를 통해 모델 학습에 필요한 데이터 배치를 반복적으로 얻을 수 있게 됩니다.
- mnist_train : datasets.MNIST와 transforms.Compose를 통해 불러오고 전처리까지 마친 Dataset 객체
- batch_size : DataLoader가 한 번에 몇 개의 데이터 샘플(이미지-레이블 쌍)을 묶어서 모델에 전달할지 결정
- shuffle : 데이터셋의 순서를 무작위로 섞을지(True) 여부를 결정
"""
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
"""
DataLoader의 역할 요약:
DataLoader는 다음과 같은 중요한 작업을 자동으로 처리해 줍니다.

데이터 묶기 (Batching): 개별 샘플들을 지정된 batch_size 단위로 묶어줍니다.

데이터 섞기 (Shuffling): 각 에포크마다 데이터의 순서를 무작위로 섞어줍니다 (shuffle=True일 때).

다중 프로세스 로딩 (Multi-process Loading): (이 코드에는 명시되어 있지 않지만, num_workers 인자를 사용하면) 백그라운드에서 여러 개의 
프로세스를 사용하여 데이터를 미리 로드함으로써 GPU가 데이터를 기다리는 시간을 줄여 학습 속도를 높일 수 있습니다.
"""