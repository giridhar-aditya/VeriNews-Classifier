# 📰 NewsGuard-ML

## 🔍 Overview

**NewsGuard-ML** is a machine learning project that classifies news articles as **real** or **fake** using a deep learning model trained on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/code) by Clément Bisaillon. The model achieves an impressive **99% accuracy** on the test set, making it highly effective at spotting credible vs misleading news content.

## 📚 Dataset

This project uses the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/code), containing:

- 📰 **Fake.csv**: 23,502 fake news articles  
- 🗞️ **True.csv**: 21,417 real news articles  

Each article is labeled **real** or **fake**, providing a solid base for training and evaluation.

## 🧠 Model Architecture

The model uses:

- ✍️ **TF-IDF Vectorization** to convert text into numerical features  
- 🔗 Fully connected neural network layers to learn complex patterns  
- 🛡️ Dropout regularization to prevent overfitting during training  

## 📊 Performance Metrics

On the test set, the model delivers:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 99% ✅ |
| Precision | 99% ✅ |
| Recall    | 99% ✅ |
| F1-Score  | 99% ✅ |

This shows excellent performance in detecting fake news while minimizing errors.

## 🚀 Usage

1. Clone this repository  
2. Install the required dependencies  
3. Load the saved model and vectorizer  
4. Input news text to get a **real** or **fake** prediction  
