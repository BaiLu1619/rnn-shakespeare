# Character-RNN

## 项目介绍

Character-RNN 是一个基于 PyTorch 的字符级文本生成项目。它从文本构建字符
词表，训练循环神经网络预测下一个字符，再根据提示文本逐字符生成新文本。
项目使用 Tiny Shakespeare 训练基础 `tanh` RNN，展示隐藏状态更新和字符级
自回归生成的完整流程。

## 功能特性

- 自动下载并校验 Tiny Shakespeare
- 自动构建字符词表，将字符转换为整数 token
- 使用 `Embedding + tanh RNN + Linear` 实现字符级语言模型
- 按时间顺序划分训练集、验证集和测试集
- 将文本整理为连续 batch stream，在相邻序列块之间传递隐藏状态
- 使用截断误差反向传播（truncated BPTT）训练
- 使用 NLL 和 perplexity 评价模型
- 支持提示文本（priming）和采样温度（temperature）
- 自动优先使用 CUDA 或 Apple Metal GPU，无可用 GPU 时使用 CPU
- 保存验证集表现最好的模型，固定随机种子保证结果稳定

## 安装

环境要求：Python 3.10、3.11 或 3.12。

```bash
conda create -n character-rnn python=3.10 -y
conda activate character-rnn
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 数据准备

Tiny Shakespeare 是一个约 1.1 MB 的英文文本数据集，包含莎士比亚多部戏剧中的
约 100 万个字符。项目将全部不同字符组成词表，并按文本原有顺序划分为：

- 90% 训练集
- 5% 验证集
- 5% 测试集

下载并校验数据：

```bash
python main.py --download-data
```

数据保存为 `data/tinyshakespeare/input.txt`。直接开始训练时，如果本地没有数据，
程序也会自动下载。

## 使用方法

在项目根目录运行：

```bash
python main.py
```

该命令会一次完成整个流程：

```text
加载数据 → 构建字符词表 → 训练 → 验证 → 测试 → 生成文本
```

使用其他配置文件：

```bash
python main.py --config path/to/config.yaml
```

## 参数配置

默认配置位于 [`configs/config.yaml`](configs/config.yaml)：

```yaml
data_dir: data/tinyshakespeare
sequence_length: 100
train_fraction: 0.90
validation_fraction: 0.05
embedding_dim: 128
hidden_dim: 256
num_layers: 1
dropout: 0.0
epochs: 20
batch_size: 64
learning_rate: 0.002
weight_decay: 0.0
grad_clip: 1.0
seed: 42
prompt: "ROMEO:"
generation_length: 500
temperature: 0.8
```

主要参数：

- `sequence_length`：每个序列块包含的字符数，也是截断 BPTT 的最大长度
- `batch_size`：同时处理的连续字符流数量
- `embedding_dim`：字符 embedding 的维度
- `hidden_dim`：RNN 隐藏状态的维度
- `num_layers`：RNN 层数
- `epochs`：训练轮数
- `grad_clip`：梯度裁剪阈值，用于缓解 RNN 梯度爆炸
- `prompt`：生成文本的开头，字符必须存在于数据集词表中
- `generation_length`：在提示文本之后生成的字符数
- `temperature`：采样温度；越低越保守，越高越随机，必须大于0

## 模型原理

### 自回归分解

对于字符序列 $x_1,x_2,\ldots,x_T$，Character RNN 将联合概率分解为：

```math
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p\left(x_t\mid x_{<t}\right)
```

训练数据使用错开一位的输入和目标。例如，输入 `hell` 时，目标为 `ello`，模型在
每个时间步预测下一个字符。编码后的文本被排列成多条连续字符流，每次读取
`sequence_length` 个时间步。

### 隐藏状态

字符 $x_t$ 首先通过 embedding 层转换为向量 $e_t$。基础 RNN 使用固定维度的隐藏
状态总结之前的字符：

```math
h_t=\tanh\left(W_{xh}e_t+W_{hh}h_{t-1}+b_h\right)
```

其中 $h_{t-1}$ 表示之前的历史信息，$h_t$ 表示加入当前字符后的新历史信息。所有
时间步共享同一组 RNN 参数，因此参数量不会随序列长度增加。

### 下一个字符预测

输出层将隐藏状态映射为词表中每个字符的 logits：

```math
o_t=W_{hy}h_t+b_y
```

Softmax 将 logits 转换为下一个字符的条件概率：

```math
p\left(x_{t+1}=k\mid x_{\le t}\right)
=\frac{\exp\left(o_t^{(k)}\right)}
{\sum_{j=1}^{K}\exp\left(o_t^{(j)}\right)}
```

其中 $K$ 是字符词表大小。RNN 按时间顺序更新隐藏状态，但多条字符流可以组成 batch
并行计算。隐藏状态会从当前序列块传递给下一块，同时在块边界调用 `detach()` 截断
梯度，因此反向传播最长为 `sequence_length` 个时间步。

生成时未来字符未知，模型先用 `prompt` 更新隐藏状态，再逐字符采样并将结果重新输入
RNN。温度参数通过缩放 logits 控制生成文本的确定性和多样性。

## 评价指标

- `NLL`：真实下一个字符的平均负对数似然，越低越好
- `perplexity`：`exp(NLL)`，表示模型在每个位置的平均不确定程度，越低越好

测试集只用于训练结束后的最终评价。每轮训练根据验证集 NLL 选择最佳模型。

## 项目结构

```text
rnn-shakespeare/
├── configs/
│   └── config.yaml          # 模型、训练和生成参数
├── data/
│   └── README.md            # 数据下载说明
├── model/
│   └── rnn.py               # Character RNN模型
├── util/
│   ├── config.py            # 配置、设备和随机种子
│   └── data_loader.py       # 字符编码与连续batch stream
├── train.py                 # 训练、验证和测试
├── sample.py                # 提示文本与温度采样
├── main.py                  # 完整流程入口
├── requirements.txt
└── README.md
```

模型、数据、训练和采样功能分别放在独立模块中，`main.py` 作为统一运行入口。

## 输出结果

训练期间终端会输出训练集和验证集 NLL，以及验证集 perplexity。训练完成后输出测试集
NLL 和 perplexity，并生成：

- `rnn_model.pt`：验证集 NLL 最低的模型、字符词表和配置
- `generated_text.txt`：从 `prompt` 开始生成的完整文本

## 参考资料

- Character RNN：[karpathy/char-rnn](https://github.com/karpathy/char-rnn)
- PyTorch：[RNN 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- Andrej Karpathy：[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
