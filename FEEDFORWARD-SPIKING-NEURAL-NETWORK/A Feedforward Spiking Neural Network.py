# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title="Spiking Neural Network"):
    """
    3계층 스파이킹 신경망의 스파이크 트레인을 시각화하는 함수.

    Args:
        spk_in (torch.Tensor): 입력 스파이크 트레인 (시간, 배치, 입력 뉴런 수)
        spk1_rec (torch.Tensor): 은닉 계층 스파이크 트레인 (시간, 배치, 은닉 뉴런 수)
        spk2_rec (torch.Tensor): 출력 계층 스파이크 트레인 (시간, 배치, 출력 뉴런 수)
        title (str): 그래프의 제목
    """

    # 텐서 형태를 (시간, 뉴런 수)로 변환 (배치 차원 제거)
    # 배치 크기가 1이므로 unsqueeze()로 추가했던 차원을 squeeze()로 제거합니다.
    spk_in = spk_in.squeeze(1).cpu().detach().numpy()
    spk1_rec = spk1_rec.squeeze(1).cpu().detach().numpy()
    spk2_rec = spk2_rec.squeeze(1).cpu().detach().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0.1})
    fig.suptitle(title, fontsize=16)

    # 1. 입력 스파이크 그래프 (Input Spikes)
    ax1 = axes[0]
    num_inputs = spk_in.shape[1]  # 입력 뉴런 수
    # 스파이크가 발생한 위치를 (시간, 뉴런 인덱스)로 변환
    t_in, neuron_in = torch.where(torch.from_numpy(spk_in) == 1)
    ax1.scatter(t_in, neuron_in, s=2, alpha=0.6, color='k')
    ax1.set_ylabel(f"Input Spikes\n({num_inputs})", fontsize=10)
    ax1.set_ylim([-0.5, num_inputs - 0.5])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. 은닉 계층 스파이크 그래프 (Hidden Layer Spikes)
    ax2 = axes[1]
    num_hidden = spk1_rec.shape[1]  # 은닉 뉴런 수
    t_hidden, neuron_hidden = torch.where(torch.from_numpy(spk1_rec) == 1)
    ax2.scatter(t_hidden, neuron_hidden, s=2, alpha=0.6, color='k')
    ax2.set_ylabel(f"Hidden Layer\n({num_hidden})", fontsize=10)
    ax2.set_ylim([-0.5, num_hidden - 0.5])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. 출력 스파이크 그래프 (Output Spikes)
    ax3 = axes[2]
    num_outputs = spk2_rec.shape[1]  # 출력 뉴런 수
    t_out, neuron_out = torch.where(torch.from_numpy(spk2_rec) == 1)
    ax3.scatter(t_out, neuron_out, s=10, alpha=0.8, color='k', marker='|')  # 출력은 점 대신 막대로 시각화
    ax3.set_ylabel(f"Output Spikes\n({num_outputs})", fontsize=10)
    ax3.set_ylim([-0.5, num_outputs - 0.5])
    ax3.set_xlabel("time step", fontsize=12)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 제목 공간 확보
    plt.show()

def plot_cur_mem_spk(cur, mem, spk, thr_line=1, ylim_max1=0.5, title="LIF Neuron Model"):
    """
    LIF 뉴런 모델의 시뮬레이션 결과를 시각화하는 함수.

    Args:
        cur (torch.Tensor): 입력 전류 (Input Current) 텐서.
        mem (torch.Tensor): 막 전위 (Membrane Potential) 텐서.
        spk (torch.Tensor): 출력 스파이크 (Output Spike) 텐서.
        thr_line (float): 막 전위 그래프에 표시할 임계값 (Threshold) 선.
        ylim_max1 (float): 입력 전류 그래프의 Y축 최대값.
        title (str): 그래프의 제목.
    """

    # 1. figure와 axes 설정 (총 3개의 서브플롯)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0.1})
    fig.suptitle(title, fontsize=16)

    # 2. 입력 전류 그래프 (Input Current)
    ax1 = axes[0]
    ax1.plot(cur.cpu().numpy(), color='tab:orange')
    ax1.set_ylabel("Input Current ($I_{in}$)", fontsize=12)
    ax1.set_ylim([0, ylim_max1])
    ax1.set_xlim([0, len(cur)])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # 3. 막 전위 그래프 (Membrane Potential)
    ax2 = axes[1]
    ax2.plot(mem.cpu().numpy(), color='tab:blue')
    ax2.axhline(y=thr_line, linestyle='--', color='gray')  # 임계값 선 추가
    ax2.set_ylabel("Membrane Potential ($U$)", fontsize=12)
    ax2.set_ylim([0, 1.2])  # 임계값보다 약간 높게 y축 설정
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # 4. 출력 스파이크 그래프 (Output Spike)
    ax3 = axes[2]
    # 스파이크가 발생한 시점에 선을 그립니다.
    spk_times = torch.where(spk.cpu() == 1)[0]
    ax3.vlines(spk_times, ymin=0, ymax=1, color='tab:gray', linewidth=2)
    ax3.set_ylabel("Output Spike ($S$)", fontsize=12)
    ax3.set_xlabel("time step", fontsize=12)
    ax3.set_yticks([0, 1])
    ax3.set_ylim([-0.1, 1.1])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.tick_params(left=False)

    plt.show()

# Layer parameters
num_inputs = 784
num_hidden = 1000
num_outputs = 10
beta = 0.99

# Initialize layers
fc1 = nn.Linear(num_inputs, num_hidden)
lif1 = snn.Leaky(beta=beta)
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snn.Leaky(beta=beta)

# Initialize hidden states
mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

# record outputs
mem2_rec = []
spk1_rec = []
spk2_rec = []

spk_in = snn.spikegen.rate_conv(torch.rand((200, 784))).unsqueeze(1)
print(f"Dimensions of spk_in: {spk_in.size()}")
# >>> Dimensions of spk_in: torch.Size([200, 1, 784])

num_steps = 200

# network simulation
for step in range(num_steps):
    cur1 = fc1(spk_in[step]) # post-synaptic current <-- spk_in x weight
    spk1, mem1 = lif1(cur1, mem1) # mem[t+1] <--post-syn current + decayed membrane
    cur2 = fc2(spk1)
    spk2, mem2 = lif2(cur2, mem2)

    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)

# convert lists to tensors
mem2_rec = torch.stack(mem2_rec)
spk1_rec = torch.stack(spk1_rec)
spk2_rec = torch.stack(spk2_rec)

plot_snn_spikes(spk_in, spk1_rec, spk2_rec, "Fully Connected Spiking Neural Network")

# from IPython.display import HTML

fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
spk2_rec = spk2_rec.squeeze(1).detach().cpu()

import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'

#  Plot spike count histogram
anim = splt.spike_count(spk2_rec, fig, ax, labels=labels, animate=True)
# HTML(anim.to_html5_video())
anim.save(".\\spike_bar.gif")

# plot membrane potential traces
splt.traces(mem2_rec.squeeze(1), spk=spk2_rec.squeeze(1))
fig = plt.gcf()
fig.set_size_inches(8, 6)