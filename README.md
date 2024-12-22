# Movie Genre Prediction: Comparative Analysis of BERT, RNN, and Naive Bayes

## Project Description
This project applies **Natural Language Processing (NLP)** techniques to predict movie genres based on plot summaries from the IMDb dataset. Conducted as part of a university course on NLP and Text Analytics, this work compares the performance of three models:
1. **BERT** (Bidirectional Encoder Representations from Transformers)
2. **Recurrent Neural Network (RNN)** with LSTM architecture
3. **Multinomial Naive Bayes (MNB)**

Developed collaboratively by a group of four students, this project evaluates each model's strengths, limitations, and practical implications.

---

## Overview
Movie genres are traditionally classified manually or using simple keyword-based systems. However, these methods struggle to handle ambiguities, contextual relationships, and evolving genre definitions. This project explores modern NLP techniques to automate genre classification while balancing computational efficiency and prediction accuracy.

**Research Question**:  
*How can movie genres be predicted from plot summaries using NLP techniques, and which model is best suited for accurate classification?*

---

## Dataset
The project utilizes the **IMDb Genre Classification Dataset**, sourced from Kaggle. It includes 54,214 training samples and 54,200 test samples, with plot summaries labeled by genre.

### Data Preprocessing
1. **Deduplication**: Removed duplicate entries.
2. **Language Filtering**: Retained only English-language descriptions.
3. **Short Description Removal**: Excluded summaries with fewer than 130 characters.
4. **Tokenization**: Split text into individual tokens, removed stop words, and applied stemming.
5. **Class Balancing**: Addressed imbalanced genres using **SMOTE** (Synthetic Minority Over-sampling Technique).

---

## Models and Methods

### 1. **Multinomial Naive Bayes (Baseline)**
- **Text Vectorization**: Used TF-IDF with unigrams, bigrams, and trigrams.
- **Class Balancing**: Employed SMOTE.
- **Hyperparameter Tuning**: GridSearchCV to optimize the smoothing parameter (`alpha`).
- **Accuracy**: **53.38%**.

### 2. **Recurrent Neural Network (RNN)**
- **Architecture**: LSTM with embedding layers, dropout layers, and a dense layer with softmax activation.
- **Optimization**: Adam optimizer with categorical cross-entropy loss.
- **Accuracy**: **53.51%**.

### 3. **BERT (Bidirectional Encoder Representations from Transformers)**
- **Pretrained Model**: Fine-tuned `BertForSequenceClassification`.
- **Tokenizer**: Prepared textual data into token IDs.
- **Training**: Hugging Face `Trainer` class for efficient fine-tuning.
- **Accuracy**: **61%**.

---

## Results
| Model        | Accuracy  | Precision | Recall | F1-Score |
|--------------|-----------|-----------|--------|----------|
| Naive Bayes  | 53.38%    | 0.51      | 0.53   | 0.52     |
| RNN          | 53.51%    | 0.54      | 0.54   | 0.54     |
| **BERT**     | **61%**   | 0.58      | 0.61   | 0.59     |

**Key Insight**:  
BERT outperformed the other models in terms of accuracy and F1-score, but it is computationally intensive. RNN and Naive Bayes provide faster, less resource-intensive alternatives.


