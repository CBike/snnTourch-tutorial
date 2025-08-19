import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset


sensor_size = tonic.datasets.NMNIST.sensor_size


frame_transform = transforms.Compose([
    transforms.Denoise(filter_time=10000),
    transforms.ToFrame(sensor_size=sensor_size,
                       time_window=1000),
])


trainset = tonic.datasets.NMNIST(save_to='./data',  transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)

# 데이터 로딩 속도를 높이기 위해, 디스크 캐싱과 배치 처리

from torch.utils.data import DataLoader
from tonic import DiskCachedDataset

# 데이터셋을 디스크에 캐싱
cached_trainset = DiskCachedDataset(trainset, cache_path= './cache/nmnist/train')
cached_dataloader = DataLoader(cached_trainset)

batch_size = 128
train_loader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

# 배치크기와 콜레이션 함수를 이용한 데이터로더 정의

def load_sample_batched():
    events, target = next(iter(cached_dataloader))

# 메인 메모리 캐싱
from tonic import MemoryCachedDataset

cached_trainset = MemoryCachedDataset(trainset)


#  캐싱 래퍼(caching wrappers)와 데이터로더(dataloaders)를 정의
import torch
import torchvision

transform = transforms.Compose([
    torch.from_numpy,
    torchvision.transforms.RandomRotation([-10, 10])
])

cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')
cached_testset = DiskCachedDataset(testset, cache_path='./cache/nminst/test')

batch_size = 128
train_loader  = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
test_loader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_size))

# Define our Network
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn as nn

# 사용 가능한 장치 (GPU 또는 CPU)를 확인하여 할당
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


#뉴런 및 시뮬레이션 매게변수
spike_grad = surrogate.atan() #대리 경사 함수로 아크탄젠트 사용
beta = 0.5 # 막 전위 붕괴율


# 네트워크 초기화

net = nn.Sequential(nn.Conv2d(2, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 32, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(32*5*5, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)).to(device)


def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)

    for step in range(data.size(0)):
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)


    return torch.stack(spk_rec)


optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)


num_epochs = 1
num_iters = 50
loss_hist = []
acc_hist = []


#training Loop

# training Loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device).long()

        net.train()
        spk_rec = forward_pass(net, data)

        # ✨ 이 부분에 아래 코드를 추가하세요 ✨

        # 1. targets를 원-핫 인코딩
        num_outputs = 10
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_outputs).float()
        targets_one_hot = targets_one_hot.to(device)

        # 2. spk_rec를 시간 차원에 대해 합산하여 총 스파이크 수 계산
        spike_count = spk_rec.sum(dim=0)

        # 3. 손실 함수에 수정된 텐서들을 전달
        loss_val = loss_fn(spike_count, targets_one_hot)

        # Gradient Calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # store loss history for future plotting
        loss_hist.append(loss_val.item())
        print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

        # Accuracy calculation also needs the aggregated spike count
        acc = SF.accuracy_rate(spike_count, targets)  # accuracy_rate 함수에도 spike_count를 전달
        acc_hist.append(acc.item())
        print(f"Accuracy: {acc * 100:.2f}%\n")

        if i == num_iters:
            break

import matplotlib.pyplot as plt

fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()
