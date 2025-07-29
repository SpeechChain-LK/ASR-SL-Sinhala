# 🎤 Sinhala ASR (Automatic Speech Recognition) Project

A comprehensive research project for developing Automatic Speech Recognition models specifically for the Sinhala language (සිංහල) using OpenAI's Whisper architecture.

## 📋 Project Overview

This project focuses on training and evaluating Whisper-based ASR models for Sinhala speech recognition. The repository contains multiple experiments, data processing pipelines, and model evaluation notebooks designed to improve Sinhala speech-to-text conversion accuracy.

### 🎯 Key Features

- **Multi-scale Training**: Experiments with datasets of varying sizes (10K, 15K, 50K samples)
- **Whisper Integration**: Fine-tuning OpenAI's Whisper models for Sinhala language
- **Comprehensive Evaluation**: Performance testing and model comparison across different configurations
- **Data Processing Pipeline**: Complete workflow from raw audio to clean training data

## 📁 Project Structure

```
ASR-SL-Sinhala/
├── 📊 Data Processing
│   ├── combine_train_test_data.ipynb          # Data combination utilities
│   ├── final-data-combination.ipynb          # Final dataset preparation
│   └── combined_file_path_sentence.csv       # Combined dataset file
│
├── 🧪 Experiments
│   ├── EXP 01/                               # Initial experiments
│   │   ├── sinhala-asr-model-tester.ipynb   # Model testing
│   │   ├── sinhala-asr-whisper EXP 01.ipynb # First training experiment
│   │   └── test_audio.wav                    # Sample audio file
│   ├── EXP 02/                               # Second experiment iteration
│   ├── EXP 03/                               # Third experiment with variations
│   └── EXP 04/                               # Latest experiment
│
├── 📈 Training & Testing
│   ├── whisper_sample_training.ipynb         # Sample training (100 records)
│   └── sinhala-asr-model-tester.ipynb       # Model evaluation
│
└── 📂 processed_asr_data/                    # Processed datasets
    ├── Dataset variants (10K, 15K, 50K samples)
    ├── Train/test splits
    ├── Metadata and vocabulary files
    └── Clean data manifests
```

## 📊 Dataset Information

### Dataset Statistics
- **Total Samples**: 90,194 audio-text pairs
- **Language**: Sinhala (සිංහල)
- **Format**: CSV with sentence and file path columns
- **Audio Format**: FLAC files
- **Data Source**: Large Sinhala ASR Training Dataset from Kaggle

### Data Source
This project utilizes the **Large Sinhala ASR Training Dataset** available on Kaggle:
- **Dataset URL**: [https://www.kaggle.com/datasets/keshan/large-sinhala-asr-training-dataset](https://www.kaggle.com/datasets/keshan/large-sinhala-asr-training-dataset/data)
- **Creator**: Keshan
- **Description**: A comprehensive collection of Sinhala speech-text pairs specifically designed for ASR model training
- **Quality**: Pre-processed and cleaned Sinhala audio data with corresponding transcriptions

### Dataset Variants
- **10K Dataset**: 10,000 samples for quick experiments
- **15K Dataset**: 15,000 samples for medium-scale training
- **50K Dataset**: 50,000 samples for comprehensive training
- **Full Dataset**: 90,194 samples for maximum performance

## 🚀 Getting Started

### Prerequisites

```bash
pip install transformers datasets torch torchaudio librosa soundfile evaluate jiwer accelerate tensorboard pandas numpy scikit-learn
```

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ImeshaDilshani/ASR-SL-Sinhala.git
   cd ASR-SL-Sinhala
   ```

2. **Set up the environment**
   ```bash
   pip install -r requirements.txt  # Create this file with dependencies above
   ```

3. **Run a sample training**
   ```bash
   jupyter notebook whisper_sample_training.ipynb
   ```

## 🧪 Experiments

### Experiment 01
- **Focus**: Initial Whisper model fine-tuning
- **Dataset**: Small subset for proof of concept
- **Model**: whisper-base

### Experiment 02
- **Focus**: Scaling up training data
- **Improvements**: Enhanced data preprocessing

### Experiment 03
- **Focus**: Model architecture variations
- **Variations**: Two different approaches tested

### Experiment 04 (Latest)
- **Focus**: Optimized training pipeline
- **Features**: 
  - Improved data loading
  - Enhanced evaluation metrics
  - Better model checkpointing

## 📈 Model Performance

The project includes comprehensive evaluation metrics:
- **WER (Word Error Rate)**: Primary metric for ASR evaluation
- **CER (Character Error Rate)**: Character-level accuracy
- **BLEU Score**: Translation quality metric
- **Custom Sinhala metrics**: Language-specific evaluation

## 🔧 Data Processing Pipeline

1. **Data Collection**: Gathering Sinhala audio-text pairs
2. **Cleaning**: Text normalization and audio preprocessing
3. **Splitting**: Train/test/validation splits
4. **Manifest Creation**: JSONL format for training
5. **Vocabulary Building**: Sinhala character/word vocabularies

## 📝 Usage Examples

### Training a Model
```python
# Load the training notebook
jupyter notebook EXP\ 04/sinhala-asr-whisper-EXP-04.ipynb
```

### Testing a Model
```python
# Load the testing notebook
jupyter notebook sinhala-asr-model-tester.ipynb
```

### Data Processing
```python
# Process and combine datasets
jupyter notebook final-data-combination.ipynb
```

## 🎯 Model Architecture

- **Base Model**: OpenAI Whisper (whisper-base)
- **Fine-tuning**: Supervised fine-tuning on Sinhala data
- **Architecture**: Encoder-Decoder Transformer
- **Input**: Mel-spectrogram features
- **Output**: Sinhala text sequences

## 📊 Results and Evaluation

Each experiment folder contains detailed results and evaluation metrics. Key performance indicators include:

- Model convergence plots
- WER/CER across different dataset sizes
- Inference time benchmarks
- Memory usage analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of S2S Research and is available for academic and research purposes.

## 🙏 Acknowledgments

- **Keshan** for providing the [Large Sinhala ASR Training Dataset](https://www.kaggle.com/datasets/keshan/large-sinhala-asr-training-dataset/data) on Kaggle
- OpenAI for the Whisper architecture
- Hugging Face for the Transformers library
- The Sinhala speech recognition research community
- Contributors to Sinhala language processing tools
- Kaggle community for hosting and maintaining the dataset

## 📧 Contact

For questions and collaboration opportunities, please reach out through the repository issues or contact the research team.

---

**Project Status**: Active Development
**Last Updated**: July 29, 2025
**Research Focus**: Speech-to-Speech Translation Research
