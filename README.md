# CLoGNet: Channel-Local–Global Network for Image Classification

Official implementation of **CLoGNet**, a lightweight deep neural network that integrates **local spatial features, global contextual information, and channel attention** for robust image representation.

CLoGNet combines three complementary feature encoders:

- **Local Encoder** – captures fine-grained spatial details using depthwise separable convolutions  
- **Global Encoder** – models long-range spatial dependencies through attention-based global context aggregation  
- **Channel Encoder** – learns channel-wise importance using adaptive channel attention  

These representations are then **feature-interacted and adaptively fused**, allowing the network to balance local texture, global structure, and channel importance.

---

# Architecture Overview

CLoGNet consists of the following components:

1. **Multi-Scale Stem**
   - Parallel convolution branches (1×1, 3×3, 5×5)
   - Extracts multi-scale low-level features

2. **CLG Fusion Blocks**
   - Local Encoder
   - Global Encoder
   - Channel Encoder
   - Cross-interaction module
   - Adaptive fusion weights

3. **Classification Head**
   - Global Average Pooling
   - Fully connected layers
   - Dropout regularization

The architecture is designed to maintain **high representational power with relatively low computational complexity**.

---

# Model Architecture

```
Input Image
     │
     ▼
MultiScaleStem
     │
     ▼
CLG Fusion Blocks
 ├── Local Encoder
 ├── Global Encoder
 └── Channel Encoder
     │
Cross Interaction
     │
Adaptive Fusion
     │
Residual Connection
     ▼
Classification Head
     ▼
Output Prediction
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/clognet.git
cd clognet
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- PyTorch  
- TorchVision  
- Scikit-learn  
- Matplotlib  

---

# Usage

## Initialize Model

```python
from model import CLoGNet

model = CLoGNet(
    in_channels=3,
    num_classes=2,
    widths=[32, 48, 80],
    blocks=[1,1,1]
)
```


---

# Key Components

### MultiScaleStem
Extracts features at different receptive fields using parallel convolution paths.

### Local Encoder
Captures fine spatial details using **depthwise separable convolutions**.

### Global Encoder
Aggregates global contextual information via pooled attention.

### Channel Encoder
Learns channel dependencies through lightweight channel attention.

### Cross Interaction Module
Allows local, global, and channel features to influence each other.

### Adaptive Fusion
Uses learnable weights to dynamically combine representations.

---

# Advantages of CLoGNet

- Lightweight architecture  
- Multi-scale feature extraction  
- Joint modeling of spatial and channel relationships  
- Adaptive feature fusion  
- Suitable for resource-constrained environments  

---

# Example Applications

CLoGNet can be applied to:

- Image classification
- Facial analysis
- Medical image classification
- Lightweight vision systems
- Edge AI deployment

---


This work is implemented using the PyTorch deep learning framework.
