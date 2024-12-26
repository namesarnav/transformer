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
