# Hate Speech Detection Project

## Overview

This project implements and compares two distinct approaches for detecting hate speech and offensive language in social media text. The goal is to classify tweets into three categories: **Hate Speech**, **Offensive Language**, and **Neither**.

The project explores two different model architectures:
1.  **LSTM (Long Short-Term Memory)**: A deep learning approach using word embeddings and recurrent neural networks.
2.  **HateBERT**: A transfer learning approach using a BERT model pre-trained on abusive language communities (Reddit).

## Key Features

*   **Data Augmentation**: Generates synthetic non-offensive sentences to address class imbalance and reduce bias.
*   **Advanced Preprocessing**: Includes text cleaning, URL removal, lemmatization, and handling of social media specific artifacts (mentions, hashtags).
*   **Class Balancing**: Utilizes SMOTE (for LSTM) and weighted loss functions (for HateBERT) to handle the skewed distribution of classes.
*   **Model Comparison**: Provides a comparison between a traditional deep learning model (LSTM) and a state-of-the-art transformer model (HateBERT).

## Dataset

The project uses the **[Hate Speech and Offensive Language Dataset](https://huggingface.co/datasets/tdavidson/hate_speech_offensive)** by Thomas Davidson et al.

*   **Class 0**: Hate Speech
*   **Class 1**: Offensive Language
*   **Class 2**: Neither

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    The notebooks contain cells to install the required packages. Generally, you will need:
    ```bash
    pip install torch transformers datasets scikit-learn pandas numpy tqdm imbalanced-learn spacy tweet-preprocessor nlpaug
    python -m spacy download en_core_web_sm
    ```

## Usage

### 1. LSTM Approach
Open `hatespeech_Lstm.ipynb` in Jupyter Notebook or VS Code.
*   This notebook walks through the entire pipeline from data loading, augmentation, preprocessing (lemmatization, tokenization), to training an LSTM model.
*   It uses **SMOTE** for oversampling minority classes.

### 2. HateBERT Approach
Open `hatespeech_with_Hatebert.ipynb` in Jupyter Notebook or VS Code.
*   This notebook utilizes the **GroNLP/hateBERT** model.
*   It implements advanced preprocessing and a custom **Weighted Loss Trainer** to handle class imbalance.
*   It saves the fine-tuned model to the `best_hate_speech_model/` directory.

## Model Architectures

### LSTM Model
*   **Embedding Layer**: Converts word indices to dense vectors.
*   **LSTM Layers**: Two LSTM layers (100 and 50 units) to capture sequential dependencies.
*   **Dense Layer**: Output layer with Softmax activation for multi-class classification.

### HateBERT Model
*   **Base Model**: `GroNLP/hateBERT` (BERT base uncased, retrained on Reddit comments).
*   **Fine-tuning**: The model is fine-tuned on the specific dataset with a weighted cross-entropy loss to penalize misclassification of minority classes.

## Project Structure

```
├── hatespeech_Lstm.ipynb          # LSTM implementation notebook
├── hatespeech_with_Hatebert.ipynb # HateBERT implementation notebook
├── non_offensive_sentences.csv    # Generated augmented data (created by LSTM notebook)
├── best_hate_speech_model/        # Saved HateBERT model (generated after training)
├── hatebert best/                 # (Optional) Another model checkpoint folder
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore file
```

## Results

*   **LSTM**: Provides a baseline for deep learning performance. Effective but may struggle with subtle context compared to transformers.
*   **HateBERT**: Generally achieves higher F1-scores and better generalization due to its pre-training on relevant domain data (abusive language).

## Acknowledgments

*   Dataset provided by [Davidson et al. (2017)](https://github.com/t-davidson/hate-speech-and-offensive-language).
*   HateBERT model by [GroNLP](https://github.com/GroNLP/hateBERT).
