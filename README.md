# Advanced Face Recognition System

A comprehensive face recognition system built with Python, TensorFlow, and Keras that supports classification, metric learning, anti-spoofing, and emotion detection capabilities.

## Features

- **Multi-Modal Face Recognition**: Supports both classification and metric learning approaches
- **Anti-Spoofing Detection**: Built-in protection against presentation attacks using TensorFlow Lite models
- **Emotion Detection**: Real-time emotion classification using pre-trained deep learning models
- **Scalable Architecture**: Modular design supporting large-scale datasets and high-throughput processing
- **Comprehensive Evaluation**: Automated performance metrics and visualization tools
- **Cross-Platform Deployment**: Models exported in both Keras and TensorFlow Lite formats

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/advanced-face-recognition.git
cd advanced-face-recognition
```

2. Install listed dependencies.

3. Download pre-trained models (if available) or train your own models using the provided training scripts.

4. Download dataset at https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/data and rename to match project structure.

## Project Structure

```
advanced-face-recognition/
├── dataset/
│   ├── classification_data/
│   │   ├── train_data/          # Training images organized by identity
│   │   ├── val_data/            # Validation images organized by identity
│   │   └── test_data/           # Test images organized by identity
│   └── verification_data/       # Verification pairs and images
├── models/
│   ├── classification_model.keras
│   ├── classification_embedding.keras
│   ├── metric_embedding.keras
│   ├── anti_spoofing.tflite
│   └── emotion_detection.hdf5
├── images/
│   ├── registered/              # Registered face images
│   └── results/                 # Generated evaluation plots
├── main.py                      # Main application entry point
├── train.py                     # Model training script
├── evaluate.py                  # Model evaluation script
├── detect.py                    # Real-time detection script
└── models/train.ipynb           # Training notebook
```

## Usage

### Training Models

1. **Classification Model Training**:
```bash
python train.py --mode classification --data_path dataset/classification_data/
```

2. **Metric Learning Model Training**:
```bash
python train.py --mode metric --data_path dataset/verification_data/
```

### Evaluation

Run comprehensive model evaluation:
```bash
python evaluate.py --model_path models/classification_model.keras --test_data dataset/classification_data/test_data/
```

### Real-time Detection

Start the face recognition system:
```bash
python detect.py --model_path models/classification_model.keras --camera 0
```

### Interactive Training

Use the Jupyter notebook for interactive model development:
```bash
jupyter notebook models/train.ipynb
```

## Configuration

### Model Parameters

- **Classification Model**: Supports multi-class face classification
- **Metric Learning**: Implements triplet loss for face embedding learning
- **Anti-Spoofing**: Lightweight TensorFlow Lite model for liveness detection
- **Emotion Detection**: Pre-trained model for 7-class emotion classification

### Dataset Organization

The system expects datasets organized as follows:

- **Classification Data**: Images organized by identity folders
- **Verification Data**: Paired images for metric learning
- **Test Data**: Separate test sets for evaluation

## Performance Metrics

The system provides comprehensive evaluation metrics:

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Verification**: ROC curves, EER (Equal Error Rate), TAR/FAR curves
- **Anti-Spoofing**: Attack Presentation Classification Error Rate (APCER)
- **Emotion Detection**: Confusion matrices and per-class accuracy

## Visualization

Automated generation of performance plots:

- ROC curves for verification tasks
- Confusion matrices for classification
- Training loss and accuracy curves
- Embedding visualizations (t-SNE, PCA)

## Security Features

- **Anti-Spoofing**: Detects presentation attacks using liveness detection
- **Multi-Factor Authentication**: Combines face recognition with additional security measures
- **Secure Model Storage**: Encrypted model weights and secure deployment options

## Deployment

### Local Deployment

1. Train or download pre-trained models
2. Configure model paths in `main.py`
3. Run the application with appropriate parameters

## Acknowledgments

- TensorFlow and Keras communities for excellent deep learning frameworks
- OpenCV for computer vision capabilities
- Research community for face recognition algorithms and datasets

---

**Note**: This system is designed for research and educational purposes. For production use, ensure compliance with privacy regulations and implement appropriate security measures. 