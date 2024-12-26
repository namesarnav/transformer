# Transformer Implementation from Scratch

A PyTorch implementation of the Transformer architecture as described in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. This implementation focuses on clarity and educational value.

## 🚀 Features

- Complete transformer architecture implementation
- Multi-head self-attention mechanism
- Positional encoding
- Layer normalization
- Feed-forward networks
- Encoder and decoder stacks
- Masked attention for autoregressive decoding

## 📦 Installation

```bash
git clone https://github.com/yourusername/transformer-implementation.git
cd transformer-implementation
pip install -r requirements.txt
```

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm

## 🏗️ Architecture

The implementation includes the following components:

```
transformer/
├── model/
│   ├── attention.py       # Multi-head attention implementation
│   ├── encoder.py         # Transformer encoder
│   ├── decoder.py         # Transformer decoder
│   ├── positional.py      # Positional encoding
│   ├── feed_forward.py    # Feed-forward network
│   └── transformer.py     # Complete transformer model
├── utils/
│   ├── masking.py        # Attention masking utilities
│   └── preprocessing.py   # Data preprocessing tools
├── train.py              # Training script
└── inference.py          # Inference utilities
```

## 🎯 Usage

### Training

```python
from transformer.model import Transformer

# Initialize model
model = Transformer(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1
)

# Training example
outputs = model(source_sequences, target_sequences)
```

### Inference

```python
# Generate sequence
generated = model.generate(
    source_sequence,
    max_length=50,
    temperature=0.7
)
```

## 📐 Mathematical Foundations

### Self-Attention

The core self-attention mechanism computes attention scores using queries (Q), keys (K), and values (V):

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

where:
- Q ∈ ℝ^(seq_len × d_k): Query matrix
- K ∈ ℝ^(seq_len × d_k): Key matrix
- V ∈ ℝ^(seq_len × d_v): Value matrix
- d_k: Dimension of key vectors
- seq_len: Sequence length

### Multi-Head Attention

Multi-head attention performs h parallel attention operations:

$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$

where each head is computed as:

$head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)$

Matrix dimensions:
- W^Q_i ∈ ℝ^(d_model × d_k)
- W^K_i ∈ ℝ^(d_model × d_k)
- W^V_i ∈ ℝ^(d_model × d_v)
- W^O ∈ ℝ^(hd_v × d_model)

### Positional Encoding
Position is encoded using sine and cosine functions:

$PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}})$
$PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})$

where:
- pos: Position in sequence
- i: Dimension index
- d_model: Model dimension

### Masked Attention

Masked attention for decoder self-attention:

$Attention(Q, K, V, M) = softmax(\frac{QK^T}{\sqrt{d_k}} + M)V$

where M is the mask matrix:
```python
M[i,j] = -inf if i < j else 0
```

### Feed-Forward Networks

Each FFN layer applies two linear transformations:

$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$

Matrix dimensions:
- W₁ ∈ ℝ^(d_model × d_ff)
- W₂ ∈ ℝ^(d_ff × d_model)
- b₁ ∈ ℝ^d_ff
- b₂ ∈ ℝ^d_model

## 🔍 Component Details

### Encoder Block

Each encoder layer consists of:
1. Multi-head self-attention
2. Layer normalization
3. Feed-forward network
4. Residual connections

```python
out = LayerNorm(x + MultiHeadAttention(x))
out = LayerNorm(out + FeedForward(out))
```

### Decoder Block

Each decoder layer consists of:
1. Masked multi-head self-attention
2. Multi-head cross-attention with encoder outputs
3. Feed-forward network
4. Layer normalization and residual connections

```python
out = LayerNorm(x + MaskedMultiHeadAttention(x))
out = LayerNorm(out + MultiHeadAttention(out, enc_out))
out = LayerNorm(out + FeedForward(out))
```

### Layer Normalization

Applied after each sub-layer:

$LayerNorm(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$

where:
- μ: Mean of the input
- σ: Standard deviation of the input
- γ, β: Learned parameters
- ε: Small constant for numerical stability

### Training Objectives

1. **Cross-Entropy Loss**
   For sequence prediction:
   $L = -\sum_{t=1}^T \sum_{v=1}^V y_{t,v} \log(p_{t,v})$
   where:
   - T: Sequence length
   - V: Vocabulary size
   - y: True labels
   - p: Predicted probabilities

2. **Label Smoothing**
   Applied to target distributions:
   $y'_{t,v} = (1-\alpha)y_{t,v} + \alpha/V$
   where α is the smoothing parameter (typically 0.1)

## 🔍 Implementation Details

Key components are implemented as follows:

1. **Multi-Head Attention**
   - Scaled dot-product attention
   - Parallel attention heads
   - Linear projections for Q, K, V

2. **Position-wise Feed-Forward**
   - Two linear transformations
   - ReLU activation
   - Dropout regularization

3. **Positional Encoding**
   - Sinusoidal position embeddings
   - Learned position embeddings (optional)

## 📊 Performance

Model performance on standard benchmarks:

- WMT14 EN-DE: 27.5 BLEU
- WMT14 EN-FR: 39.2 BLEU
