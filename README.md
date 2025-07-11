# Temporal Pianist Technique Classification

## Project Overview

This is a CS229 (Machine Learning) final project that implements a comprehensive pipeline for classifying pianist technique proficiency levels using temporal data. The project combines multiple machine learning approaches including Hidden Markov Models (HMM), Multi-Layer Perceptrons (MLP), Linear Discriminant Analysis (LDA), Gaussian Discriminant Analysis (GDA), and AdaBoost to analyze temporal patterns in pianist hand movements and classify technique proficiency.

## Project Structure

```
├── MLP/                           # Multi-Layer Perceptron implementations
│   ├── Update MLP                 # Main MLP training and evaluation
│   ├── MLP with Layer Norm        # MLP with layer normalization
│   └── Plot MLP                   # Visualization and plotting utilities
├── HMM/                           # Hidden Markov Model implementations
│   ├── HMM Model                  # Main HMM implementation with enhanced features
│   └── Make train and test csvs for HMM  # Data preprocessing for HMM
├── LDA/                           # Linear Discriminant Analysis
│   └── LDA Feature Extractor      # LDA-based feature extraction
├── GDA/                           # Gaussian Discriminant Analysis
├── Adaboost/                      # AdaBoost classifier implementations
├── Feature Enhancement/            # Feature engineering and enhancement
├── Baseline Feature Tests/         # Baseline model evaluations
├── Activity Classifier to Proficiency Lvl Classification Pipeline/
│   └── Activity and Proficieny lvl Classifier  # Multi-stage classification pipeline
├── LOAD HMM/                      # HMM model loading and inference
├── New Data/                      # Processed datasets
│   ├── TrainTest Data/            # Training and test datasets
│   ├── Sequence Data/             # Temporal sequence data
│   ├── Holdout Data/              # Holdout validation datasets
│   └── HMM train data/            # HMM-specific training data
└── requirements.txt               # Python dependencies
```

## Key Features

### 1. **Temporal Analysis with HMM**
- Implements Gaussian Hidden Markov Models for temporal pattern recognition
- Processes 21 landmark features (x, y, z coordinates) from hand tracking
- Uses robust scaling and cross-validation for model training
- Supports sequence-based probability computation

### 2. **Multi-Layer Perceptron (MLP)**
- Custom implementation with multiple activation functions (ReLU, Leaky ReLU, Sigmoid)
- Supports batch training with gradient descent
- Implements softmax output for multi-class classification
- Includes layer normalization variants

### 3. **Multi-Stage Classification Pipeline**
- **Stage 1**: K-means clustering to identify activity patterns
- **Stage 2**: AdaBoost classifiers for each activity cluster
- Combines temporal and feature-based classification approaches

### 4. **Feature Engineering**
- Linear Discriminant Analysis (LDA) for dimensionality reduction
- Gaussian Discriminant Analysis (GDA) for probabilistic modeling
- Robust scaling and outlier handling
- Temporal feature extraction from landmark sequences

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CS229-Final-Project-Temporal-Pianist-Technique-Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The project requires the following key libraries:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `hmmlearn` - Hidden Markov Models
- `torch` - PyTorch for deep learning
- `matplotlib` - Visualization
- `opencv-python` - Computer vision (for landmark detection)
- `mediapipe` - Hand tracking and landmark extraction

## Usage

### 1. Data Preprocessing
```python
# Prepare HMM training data
python "HMM/Make train and test csvs for HMM"
```

### 2. Model Training

**HMM Training:**
```python
from HMM.HMM_Model import EnhancedHMM

# Initialize and train HMM
hmm = EnhancedHMM(n_components=6, sequence_length=20)
hmm.train(train_df, validation_split=0.2)
hmm.save_model()
```

**MLP Training:**
```python
from MLP.Update_MLP import MLP

# Initialize and train MLP
mlp = MLP(layer_sizes=[input_size, hidden_size, output_size])
mlp.train(X_train, Y_train, epochs=1000, learning_rate=0.01)
```

### 3. Multi-Stage Classification
```python
# Run the complete pipeline
python "Activity Classifier to Proficiency Lvl Classification Pipeline/Activity and Proficieny lvl Classifier"
```

## Model Architecture

### Hidden Markov Model (HMM)
- **States**: 6 hidden states representing different technique patterns
- **Features**: 21 landmarks × 3 coordinates (x, y, z) = 63 features
- **Sequence Length**: 20 frames per sequence
- **Covariance**: Full covariance matrix for feature relationships

### Multi-Layer Perceptron (MLP)
- **Architecture**: Configurable layer sizes with He initialization
- **Activation**: ReLU/Leaky ReLU for hidden layers, Softmax for output
- **Training**: Mini-batch gradient descent with cross-entropy loss
- **Regularization**: Layer normalization support

### Multi-Stage Pipeline
1. **Clustering**: K-means (k=6) to identify activity patterns
2. **Classification**: AdaBoost with Decision Trees for each cluster
3. **Ensemble**: Combines cluster assignment with proficiency prediction

## Data Format

The project expects temporal data with the following structure:
- **Landmark Features**: `lm_{i}_{x/y/z}` for i=0 to 20 (21 landmarks)
- **Labels**: Proficiency levels (e.g., novice, intermediate, advanced)
- **Sequences**: Temporal sequences of 20 frames each

## Results

The project evaluates classification performance using:
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Per-class performance analysis
- **Cross-validation**: K-fold validation for robust evaluation

## Contributing

This is a CS229 final project. For questions or contributions, please contact the project author.

## License

This project is for educational purposes as part of Stanford CS229 coursework.

## Acknowledgments

- Stanford CS229: Machine Learning Course
- MediaPipe for hand tracking and landmark extraction
- HMMLearn for Hidden Markov Model implementation 
