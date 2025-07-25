import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils
from torch.utils.data import DataLoader
from snntorch import spikegen
# 이 줄은 snntorch 라이브러리 내에 있는 spikegen 모듈을 가져오는(import) 코드입니다.

# spikegen (Spike Generator): 이 모듈은 다양한 방법으로 아날로그 데이터를 스파이킹 신경망이 처리할 수 있는 스파이크 열(spike train)로
# 변환하는 함수들을 제공합니다. 율 부호화(Rate Coding) 외에도 지연 부호화(Latency Coding) 등 여러 인코딩 방식이 있습니다.


# Training Parameters
batch_size=128
data_path='/tmp/data/mnist'
num_classes = 10  # MNIST has 10 output classes

# Torch Variables
dtype = torch.float

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])


mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
subset = 10
mnist_train = utils.data_subset(mnist_train, subset)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)


# Temporal Dynamics
# 총 시간 단계의 수 정의
num_steps = 10

# create vector filled with 0.5
raw_vector = torch.ones(num_steps)*0.5
# 0.5를 곱하는 이유는 스파이킹 신경망에서 **"발화율(firing rate)" 또는 "스파이크 발생 확률"**을 나타내는 아날로그(연속적인) 입력 신호로
# 간주될 수 있다. 여기서는 모든 시간 단계에서 동일하게 0.5의 발화율을 가진다고 가정한 것


# pass each sample through a Bernoulli trial
# aw_vector의 각 원소에 대해 베르누이 시행을 수행하여 스파이크 열을 생성
# torch.bernoulli는 이 확률에 따라 스파이크를 무작위로 생성하는 과정을 담당
rate_coded_vector = torch.bernoulli(raw_vector)

num_steps = 100
raw_vector = torch.ones(num_steps)*0.5
rate_coded_vector = torch.bernoulli(raw_vector)

# Iterate through minibatches
# train_loader 객체에 대해 이터레이터(iterator)를 생성
data = iter(train_loader)
data_it, targets_it = next(data)
"""
역할: 이 코드는 전체 학습 데이터셋에서 단 하나의 미니 배치를 가져와서, 이 배치에 포함된 이미지를 스파이크 데이터로 변환하는 과정을 예시로 
보여주기 위한 것입니다. 실제 학습 루프에서는 이 부분이 for data, targets in train_loader: 와 같이 반복적으로 실행됩니다.
"""


# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)

# pikegen 모듈 안에 있는 rate 함수를 호출 -> 아날로그 데이터를 율 부호화(Rate Coding) 방식으로 스파이크 열로 변환