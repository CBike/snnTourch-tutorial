# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import matplotlib.pyplot as plt



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




lif1 = snn.Leaky(beta=0.8)

# Small step current input
w=0.21
cur_in = torch.cat((torch.zeros(10), torch.ones(190)*w), 0)
mem = torch.zeros(1)
spk = torch.zeros(1)
mem_rec = []
spk_rec = []

num_steps = 200
# neuron simulation
for step in range(num_steps):
  spk, mem = lif1(cur_in[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

plot_cur_mem_spk(cur_in, mem_rec, spk_rec, thr_line=1, ylim_max1=0.5,
                 title="snn.Leaky Neuron Model")
