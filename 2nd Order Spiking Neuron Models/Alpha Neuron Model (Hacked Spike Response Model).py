import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def plot_spk_mem_spk(spk_in, mem_rec, spk_rec, title="SNN Plot"):
    """
    Alpha 뉴런 모델의 시뮬레이션 결과를 시각화하는 함수.

    Args:
        spk_in (torch.Tensor): 입력 스파이크 텐서. (1차원)
        mem_rec (torch.Tensor): 막 전위 기록 텐서.
        spk_rec (torch.Tensor): 출력 스파이크 텐서.
        title (str): 그래프의 제목.
    """

    # 텐서를 CPU로 이동 및 NumPy 배열로 변환
    spk_in = spk_in.cpu().numpy()
    mem_rec = mem_rec.squeeze(1).detach().cpu().numpy()
    spk_rec = spk_rec.squeeze(1).detach().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0.1})
    fig.suptitle(title, fontsize=16)

    # 1. 입력 스파이크 그래프
    ax1 = axes[0]
    ax1.vlines(torch.where(torch.from_numpy(spk_in) > 0)[0], ymin=0, ymax=1, color='k', linewidth=2)
    ax1.set_ylabel("Input Spikes", fontsize=12)
    ax1.set_ylim([-0.1, 1.1])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_yticks([])

    # 2. 막 전위 그래프
    ax2 = axes[1]
    ax2.plot(mem_rec, color='tab:blue')
    ax2.axhline(y=0.5, linestyle='--', color='gray', label='Threshold')  # 임계값 0.5 표시
    ax2.set_ylabel("Membrane Potential ($U_{mem}$)", fontsize=12)
    ax2.set_ylim([-0.1, 0.6])  # y축 범위 0.6으로 조정
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # 3. 출력 스파이크 그래프
    ax3 = axes[2]
    ax3.vlines(torch.where(torch.from_numpy(spk_rec) > 0)[0], ymin=0, ymax=1, color='k', linewidth=2)
    ax3.set_ylabel("Output Spikes", fontsize=12)
    ax3.set_xlabel("Time step", fontsize=12)
    ax3.set_ylim([-0.1, 1.1])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Temporal dynamics
alpha = 0.8
beta = 0.7

num_steps = 200

# initialize neuron
lif2 = snn.Alpha(alpha=alpha, beta=beta, threshold=0.5)


# input spike: initial spike, and then period spiking
w = 0.85
spk_in = (torch.cat((torch.zeros(10), torch.ones(1), torch.zeros(89),
                     (torch.cat((torch.ones(1), torch.zeros(9)),0).repeat(10))), 0) * w).unsqueeze(1)

# initialize parameters
syn_exc, syn_inh, mem = lif2.init_alpha()
mem_rec = []
spk_rec = []

# run simulation
for step in range(num_steps):
  spk_out, syn_exc, syn_inh, mem = lif2(spk_in[step], syn_exc, syn_inh, mem)
  mem_rec.append(mem)
  spk_rec.append(spk_out)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_spk_mem_spk(spk_in, mem_rec, spk_rec, "Alpha Neuron Model With Input Spikes")
