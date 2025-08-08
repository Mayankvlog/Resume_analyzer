# Resume Analyzer - Advanced RNN Neural Network

A sophisticated resume analysis system that uses an advanced RNN neural network with 5 hidden layers and 6 activation functions to classify resume text into job categories.

## 🚀 Features

- **Advanced RNN Architecture**: 5 hidden layers with 6 different activation functions
- **TensorFlow/Keras Implementation**: Modern deep learning framework
- **MLflow Experiment Tracking**: Comprehensive experiment management
- **DVC Data Version Control**: Data pipeline versioning
- **CircleCI Continuous Integration**: Automated testing and deployment
- **Pickle Model Serialization**: Efficient model persistence
- **Streamlit Web Interface**: Interactive web application
- **Comprehensive Logging**: Detailed logging in 2 files
- **Comprehensive Preprocessing**: Advanced text cleaning and feature extraction

## 🏗️ Architecture

### Neural Network Structure

The model consists of 5 hidden layers with 6 different activation functions:

1. **Layer 1**: Bidirectional LSTM (256 units) - `tanh` activation
2. **Layer 2**: Bidirectional LSTM (128 units) - `relu` activation  
3. **Layer 3**: Bidirectional LSTM (64 units) - `sigmoid` activation
4. **Layer 4**: Bidirectional LSTM (32 units) - `softplus` activation
5. **Layer 5**: Dense layer (128 units) - `elu` activation
6. **Additional Dense**: Dense layer (64 units) - `swish` activation
7. **Output Layer**: Dense layer (num_classes) - `softmax` activation (loss function)

### Activation Functions Used

- `tanh`: Hyperbolic tangent for LSTM Layer 1
- `relu`: Rectified Linear Unit for LSTM Layer 2
- `sigmoid`: Sigmoid function for LSTM Layer 3
- `softplus`: Softplus function for LSTM Layer 4
- `elu`: Exponential Linear Unit for Dense Layer 1
- `swish`: Swish function for Dense Layer 2
- `softmax`: Softmax for output layer (loss activation function)

## 📁 Project Structure

```
resume_analyzer/
├── data/
│   └── synthetic_resumes.csv          # Training dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Text preprocessing and feature extraction
│   ├── advanced_rnn_model.py          # RNN model architecture
│   ├── training_pipeline.py           # Training pipeline with MLflow
│   └── prediction_service.py          # Prediction service
├── model/                             # Trained models and artifacts
├── metrics/                           # Performance metrics
├── logs/                              # Log files
├── .circleci/
│   └── config.yml                     # CircleCI configuration
├── dvc.yaml                          # DVC pipeline configuration
├── requirements.txt                   # Python dependencies
├── main.py                           # Main application entry point
├── resume.ipynb                      # Jupyter notebook
└── README.md                         # Project documentation
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd resume_analyzer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize DVC** (optional):
   ```bash
   dvc init
   dvc remote add origin <your-dvc-remote>
   ```

## 🚀 Usage

### Command Line Interface

```bash
# Train the model
python main.py --mode train

# Make predictions
python main.py --mode predict --resume-text "Your resume text here"

# Analyze resume features
python main.py --mode analyze --resume-text "Your resume text here"

# Run demo predictions
python main.py --mode demo

# Start Streamlit web interface
python main.py --mode streamlit
```

### Streamlit Web Interface

```bash
# Start the web application
streamlit run main.py -- --mode streamlit
```

### Jupyter Notebook

Open `resume.ipynb` in Jupyter to explore the complete analysis workflow.

### Programmatic Usage

```python
from src.prediction_service import create_prediction_service

# Create prediction service
service = create_prediction_service()

# Make prediction
result = service.predict_single_resume("Your resume text here")
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📊 Model Performance

The model is trained on synthetic resume data with the following characteristics:

- **Dataset Size**: 1,000 resumes
- **Job Categories**: 5 categories (Data Scientist, Project Manager, Marketing Specialist, UX Designer, Software Engineer)
- **Text Processing**: Advanced NLP preprocessing with lemmatization and stopword removal
- **Vocabulary Size**: 10,000 words
- **Sequence Length**: 500 tokens

### Expected Performance Metrics

- **Accuracy**: ~85-90% on test set
- **Training Time**: ~10-15 minutes on CPU
- **Model Size**: ~50-100 MB

## 🔧 Configuration

### Model Parameters

You can customize the model architecture by modifying `src/advanced_rnn_model.py`:

```python
# Model configuration
vocab_size = 10000
max_length = 500
embedding_dim = 128
num_classes = 5

# Layer configurations
lstm_units = [256, 128, 64, 32]
dense_units = [128, 64]
dropout_rates = [0.3, 0.5]
```

### Training Parameters

Modify training parameters in `src/training_pipeline.py`:

```python
# Training configuration
epochs = 50
batch_size = 32
learning_rate = 0.001
early_stopping_patience = 10
```

## 📈 MLflow Integration

The project includes comprehensive MLflow integration for experiment tracking:

```bash
# Start MLflow UI
mlflow ui

# View experiments
mlflow experiments list

# Compare runs
mlflow compare <run_id_1> <run_id_2>
```

## 🔄 DVC Pipeline

The project uses DVC for data version control:

```bash
# Run the complete pipeline
dvc repro

# Run specific stage
dvc repro train_model

# Push to remote storage
dvc push
```

## 🚀 CircleCI Integration

The project includes CircleCI configuration for continuous integration:

- **Testing**: Automated testing of all modules
- **Training**: Model training in CI environment
- **Deployment**: Automated deployment package creation

## 📝 Logging

The system uses comprehensive logging with logs stored in the `logs/` directory:

- `logs/data_preprocessing.log`: Data preprocessing operations
- `logs/model_training.log`: Model training operations
- `logs/prediction_service.log`: Prediction service operations
- `logs/main_application.log`: Main application operations
- `logs/notebook.log`: Jupyter notebook operations

## 📝 API Documentation

### ResumePreprocessor

```python
from src.data_preprocessing import ResumePreprocessor

preprocessor = ResumePreprocessor(max_length=500, vocab_size=10000)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data('data/synthetic_resumes.csv')
```

### AdvancedResumeRNN

```python
from src.advanced_rnn_model import AdvancedResumeRNN

model = AdvancedResumeRNN(vocab_size, max_length, num_classes)
model.build_model()
history = model.train(X_train, y_train, X_val, y_val)
```

### ResumePredictionService

```python
from src.prediction_service import create_prediction_service

service = create_prediction_service()
result = service.predict_single_resume("Resume text here")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- MLflow for experiment tracking
- DVC for data version control
- CircleCI for continuous integration
- Streamlit for web interface
- NLTK for natural language processing

## 📞 Support

For questions and support, please open an issue on GitHub or contact the development team.

---

**Note**: This is a demonstration project using synthetic data. For production use, ensure you have appropriate data and follow best practices for model deployment and monitoring.
