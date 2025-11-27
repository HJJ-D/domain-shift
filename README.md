# Domain Shift Training for Rice Grain Classification

A PyTorch implementation of domain-adversarial neural networks (DANN) for rice grain classification with domain adaptation capabilities.

## 📋 Project Overview

This project implements a deep learning pipeline for classifying rice grain varieties while handling domain shift issues across different scanning machines. The system uses Domain-Adversarial Neural Networks (DANN) to learn domain-invariant features, improving generalization to unseen domains.

### Key Features

- 🌾 Multi-class rice grain classification
- 🔄 Domain adaptation using DANN
- 🎯 Domain-aware data splitting for robust evaluation
- 📊 Comprehensive evaluation metrics
- 🚀 Modular and extensible codebase
- 💾 Efficient data caching with CSV

## 🏗️ Architecture

The system consists of three main components:

1. **Backbone Network**: ResNet50 (pretrained on ImageNet) for feature extraction
2. **Classification Head**: Multi-class classifier for rice grain varieties
3. **Domain Classifier**: Adversarial component for domain-invariant learning

## 📁 Project Structure

```
domain-shift/
├── src/
│   ├── config.py          # Configuration management
│   ├── dataset.py         # Dataset and data loading utilities
│   ├── models.py          # Model definitions (backbone, classifier, DANN)
│   ├── trainer.py         # Training and evaluation functions
│   └── utils.py           # Utility functions (logging, checkpointing)
├── runs/                  # Training outputs (auto-generated)
├── .gitignore
├── README.md
├── train.py
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd domain-shift

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
from src.config import Config
from src.dataset import build_records, domain_aware_split, RiceDataset
from src.models import build_backbone, ClassifierHead, DomainClassifier
from src.trainer import train_one_epoch, evaluate
from src.utils import set_seed, setup_logging

# Initialize configuration
config = Config(base_path='/path/to/your/data')
config.save()

# Set random seed
set_seed(config.seed)

# Build dataset
records, classes_all, domains_all, class_to_idx = build_records(config.input_path)

# Split data with domain awareness
train_records, val_records, test_records = domain_aware_split(
    records,
    train_domains=config.train_domains,
    val_domains=config.val_domains,
    test_domains=config.test_domains
)

# Create datasets and dataloaders
# ... (see full training script)
```

## ⚙️ Configuration

Edit `src/config.py` to customize training parameters:

```python
class Config:
    # Data parameters
    img_size = 224
    batch_size = 64
    num_workers = 8
    
    # Training parameters
    epochs = 20
    lr = 1e-4
    weight_decay = 1e-4
    
    # DANN parameters
    use_dann = True
    lambda_dann = 0.5
    
    # Domain split
    train_domains = ['848', '899', '900', '895']
    val_domains = ['901', '898', '896']
    test_domains = ['871', '882', '890']
    
    # Domain sampling (optional)
    domain_sample_ratios = {
        '848': 0.33,
        '899': 0.33,
        '900': 0.33,
        '895': 1.0
    }
```

## 📊 Dataset Format

The dataset should be organized as follows:

```
segmented/
├── class_1/
│   ├── machine_900/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── machine_901/
│       └── ...
├── class_2/
│   └── ...
└── ...
```

- Top-level folders represent class names
- Numeric folders (e.g., `900`, `901`) represent different machines/domains
- Images should be in PNG format

## 🎯 Training

### Basic Training

```bash
python train.py
```

### Training with Custom Config

```python
from src.config import Config

config = Config()
config.epochs = 30
config.use_dann = True
config.lambda_dann = 0.7
config.train_domains = ['848', '899']
# ... continue training
```

### Domain-Aware Split Strategies

1. **Specified Domains**: Manually specify train/val/test domains
2. **Automatic Split**: Automatically split remaining domains
3. **Domain Sampling**: Subsample specific domains for balanced training

## 📈 Evaluation Metrics

The system tracks:

- **Classification Metrics**: Accuracy, F1-score (weighted), Confusion matrix
- **Domain Metrics**: Domain classifier accuracy (lower is better for DANN)
- **Per-domain Performance**: Separate metrics for each machine/domain

### Output Files

Training outputs are saved to `runs/run_multi_<timestamp>/`:

```
runs/run_multi_20240115_143052/
├── config.json                    # Training configuration
├── domain_mapping.json            # Domain to ID mapping
├── class_mapping.json             # Class to ID mapping
├── split_info.json                # Dataset split information
├── training.log                   # Detailed training logs
├── training_history.json          # Epoch-wise metrics
├── best_model.pth                 # Best model checkpoint
├── final_model.pth                # Final model checkpoint
├── test_classification_report.txt # Detailed test metrics
└── test_per_domain.csv            # Per-domain test results
```

## 🔬 Domain Adaptation with DANN

### How It Works

1. **Feature Extraction**: Backbone network extracts features from images
2. **Classification**: Classification head predicts rice grain class
3. **Domain Adversarial**: Domain classifier tries to predict the domain
4. **Gradient Reversal**: Gradients are reversed to confuse domain classifier
5. **Result**: Features become domain-invariant

### Lambda Schedule

The domain adaptation strength increases during training:

```python
p = epoch / max_epochs
lambda = 2 / (1 + exp(-10 * p)) - 1
```

## 📊 Monitoring Training

### Training Logs

```
Epoch 5/20  train_loss=0.4523 train_acc=0.8234 acc_dom_train=0.4567 
            val_acc=0.7891 val_f1=0.7823 dom_train_eval=0.4321  time=45.2s
```

### Key Indicators

- **train_acc ↑**: Classification improving
- **val_f1 ↑**: Good generalization
- **dom_train_eval ↓**: Features becoming domain-invariant (for DANN)

## 🛠️ Advanced Usage

### Custom Backbone

```python
from src.models import build_backbone

backbone, feat_dim = build_backbone('resnet50', pretrained=True)
```

Supported backbones: `resnet18`, `resnet50`

### Custom Data Augmentation

Edit the transforms in your training script:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Loading Pretrained Models

```python
from src.utils import load_checkpoint

checkpoint = load_checkpoint('runs/.../best_model.pth', backbone, clf_head)
classes_all = checkpoint['classes_all']
class_to_idx = checkpoint['class_to_idx']
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Use `resnet18` instead of `resnet50`

2. **Domain Classifier Always High Accuracy**
   - Increase `lambda_dann` parameter
   - Check if domains are truly different

3. **Poor Generalization to Test Domains**
   - Ensure test domains are truly unseen
   - Increase training epochs
   - Try different domain sampling ratios

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{domain-shift-rice,
  title={Domain-Adversarial Neural Networks for Rice Grain Classification},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/domain-shift}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact [your-email@example.com].

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- ResNet architecture from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- DANN methodology from [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)
