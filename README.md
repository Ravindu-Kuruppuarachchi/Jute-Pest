# 🌾 Jute Pest Detection using Transfer Learning

An AI-based pest detection system for jute crops using fine-tuned pre-trained CNNs (VGG19 & InceptionV3) to identify 17 different pest types with high accuracy.

## 🎯 Overview

Automated jute pest identification system using transfer learning to detect pests early and help farmers make informed pest management decisions.

## ✨ Features

- Transfer learning with VGG19 & InceptionV3
- 17 pest class classification
- Data augmentation for improved accuracy
- Early stopping & learning rate scheduling
- Google Drive integration
- Comprehensive evaluation metrics

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| VGG19 | 84% | 84% | 84% | 85% |
| InceptionV3 | 97% | 97% | 97% | 97% |

## 📦 Installation

```bash
git clone https://github.com/Ravindu-Kuruppuarachchi/Jute-Pest.git
cd Jute-Pest
pip install -r requirements.txt
```

## 🚀 Quick Start

### Google Colab
1. Mount Google Drive
2. Update dataset paths
3. Run cells sequentially

### Local Machine
```bash
python jute_pest_detection.py
```

## 📁 Dataset Structure

```
Jute_Pest_Dataset_Split/
├── train/
├── val/
└── test/
```

**Dataset**: [Kaggle - Jute Pest Dataset](https://www.kaggle.com/datasets/simulhasantalukder/jutepestindentification)

## 🧠 Model Architecture

```
Pre-trained Model (ImageNet)
    ↓
Global Average Pooling
    ↓
Dropout (30%)
    ↓
Dense Output (17 classes)
```

## ⚙️ Key Parameters

```python
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
```

## 📝 Usage Example

```python
from tensorflow.keras.applications import InceptionV3
import tensorflow as tf

# Load pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Fine-tune for jute pest classification
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(17, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])
```

## 📂 Output Files

- `saved_models/vgg19_model/` - Trained VGG19 model
- `saved_models/inception_v3_model/` - Trained InceptionV3 model
- `checkpoints/` - Best model checkpoints
- `evaluation_metrics.json` - Performance metrics

## 🐛 Troubleshooting

**Eager Execution Error:**
```python
tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)
```

**Out of Memory:**
```python
BATCH_SIZE = 8
IMG_SIZE = 192
```

## 📚 References

- [JutePestDetect Research Paper](https://arxiv.org/abs/2308.05179)
- [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## 📄 License

MIT License

## 👨‍💼 Author

Ravindu Kuruppuarachchi

---

⭐ If helpful, please star the repository!
