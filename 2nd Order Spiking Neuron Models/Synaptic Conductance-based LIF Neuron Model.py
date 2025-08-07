import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec, title="SNN Plot"):
    """
    Synaptic 뉴런 모델의 시뮬레이션 결과를 시각화하는 함수.

    Args:
        spk_in (torch.Tensor): 입력 스파이크 텐서. (1차원)
        syn_rec (torch.Tensor): 시냅스 전류 기록 텐서. (2차원: 시간, 배치)
        mem_rec (torch.Tensor): 막 전위 기록 텐서. (2차원: 시간, 배치)
        spk_rec (torch.Tensor): 출력 스파이크 텐서. (2차원: 시간, 배치)
        title (str): 그래프의 제목.
    """

    # 텐서를 CPU로 이동 및 NumPy 배열로 변환
    # spk_in은 이미 1차원이므로 squeeze(1) 호출을 제거합니다.
    spk_in = spk_in.cpu().numpy()

    # syn_rec, mem_rec, spk_rec은 2차원이므로 squeeze(1)을 유지합니다.
    syn_rec = syn_rec.squeeze(1).detach().cpu().numpy()
    mem_rec = mem_rec.squeeze(1).detach().cpu().numpy()
    spk_rec = spk_rec.squeeze(1).detach().cpu().numpy()

    # 4개의 서브플롯을 1열로 생성
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0.1})
    fig.suptitle(title, fontsize=16)

    # --- 1. 입력 스파이크 그래프 (최상단) ---
    ax1 = axes[0]
    # 스파이크가 발생한 시간 인덱스를 찾아 수직선으로 표시
    ax1.vlines(torch.where(torch.from_numpy(spk_in) > 0)[0], ymin=0, ymax=1, color='k', linewidth=2)
    ax1.set_ylabel("Input Spikes", fontsize=12)
    ax1.set_ylim([-0.1, 1.1])  # y축 범위 설정
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_yticks([])  # y축 눈금 제거

    # --- 2. 시냅스 전류 그래프 ---
    ax2 = axes[1]
    # syn_rec 텐서의 값을 선 그래프로 표시
    ax2.plot(syn_rec, color='tab:blue')
    ax2.set_ylabel("Synaptic Current ($I_{syn}$)", fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)  # y축에 그리드 추가

    # --- 3. 막 전위 그래프 ---
    ax3 = axes[2]
    # mem_rec 텐서의 값을 선 그래프로 표시
    ax3.plot(mem_rec, color='tab:blue')
    # 임계값(1)을 나타내는 점선 추가
    ax3.axhline(y=1, linestyle='--', color='gray', label='Threshold')
    ax3.set_ylabel("Membrane Potential ($U$)", fontsize=12)
    ax3.set_ylim([-0.1, 1.5])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='y', linestyle='--', alpha=0.6)

    # --- 4. 출력 스파이크 그래프 (최하단) ---
    ax4 = axes[3]
    # 출력 스파이크가 발생한 시간 인덱스를 찾아 수직선으로 표시
    ax4.vlines(torch.where(torch.from_numpy(spk_rec) == 1)[0], ymin=0, ymax=1, color='k', linewidth=2)
    ax4.set_ylabel("Output Spikes", fontsize=12)
    ax4.set_xlabel("time step", fontsize=12)
    ax4.set_ylim([-0.1, 1.1])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.set_yticks([])

    # 그래프 레이아웃 조정 및 표시
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Temporal dynamics
alpha = 0.9
beta = 0.8
num_steps = 200

# Initialize 2nd-order LIF neuron
lif1 = snn.Synaptic(alpha=alpha, beta=beta)

# Periodic spiking input, spk_in = 0.2 V
w = 0.2
spk_period = torch.cat((torch.ones(1) * w, torch.zeros(9)), 0)
spk_in = spk_period.repeat(20)

# Initialize hidden states and output
syn, mem = lif1.init_synaptic()
spk_out = torch.zeros(1)
syn_rec = []
mem_rec = []
spk_rec = []

# Simulate neurons
for step in range(num_steps):
  spk_out, syn, mem = lif1(spk_in[step], syn, mem)
  spk_rec.append(spk_out.unsqueeze(0)) # unsqueeze(0) 추가
  syn_rec.append(syn.unsqueeze(0))     # unsqueeze(0) 추가
  mem_rec.append(mem.unsqueeze(0))     # unsqueeze(0) 추가

# convert lists to tensors
spk_rec = torch.stack(spk_rec)
syn_rec = torch.stack(syn_rec)
mem_rec = torch.stack(mem_rec)

plot_spk_cur_mem_spk(spk_in, syn_rec, mem_rec, spk_rec, "Synaptic Conductance-based Neuron Model With Input Spikes")