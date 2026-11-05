# Emotion Classification with Attention

A Deep Learning and NLP project that classifies text into six emotion categories using four neural network architectures, including LSTM, GRU, BiLSTM with Attention, and DistilBERT.

## Project Overview

This project explores multi-class emotion classification using the GoEmotions dataset. It compares traditional recurrent neural networks with an attention-based architecture and a fine-tuned Transformer model.

The system predicts one of six emotions from a given text:

- Joy
- Sadness
- Anger
- Fear
- Surprise
- Disgust

## Models Implemented

Four architectures were developed and evaluated:

1. **LSTM** – Long Short-Term Memory network.
2. **GRU** – Gated Recurrent Unit network.
3. **BiLSTM + Attention** – Bidirectional LSTM with an attention mechanism.
4. **DistilBERT** – Fine-tuned Transformer-based model.

The recurrent models use pretrained GloVe word embeddings.

## Dataset

**GoEmotions Dataset** by Google Research.

- Original dataset: 28 emotion labels.
- Target categories: 6 consolidated emotion classes.
- Training samples after filtering: 22,386.
- Validation samples after filtering: 2,766.
- Test samples after filtering: 2,781.

The original labels were mapped into six broader emotion categories. Neutral, unmapped, and conflicting multi-category examples were excluded to create a single-label classification task.

## Model Performance

Final evaluation results on the test set:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| DistilBERT | **84.90%** | **73.70%** |
| BiLSTM + Attention | 77.85% | 67.15% |
| GRU | 78.50% | 66.56% |
| LSTM | 76.88% | 65.38% |

**Best Performing Model: DistilBERT**

DistilBERT achieved the highest test accuracy and Macro F1 score among the four models.

## Evaluation & Visualization

The project includes:

- Accuracy and Macro F1 comparison.
- Per-class Precision, Recall, and F1-score.
- Confusion matrices.
- Attention weight visualization.
- Example predictions with confidence scores.

## Interactive Application

A Gradio-based interface allows users to:

- Select example sentences from the test dataset.
- Enter custom text.
- Predict the dominant emotion using DistilBERT.
- View confidence scores for all six emotion categories.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- GloVe Embeddings
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Gradio
- Google Colab

## Project Structure

```text
Emotion-Classification-with-Attention/
│
├── Emotion_Classification_with_Attention.ipynb
├── README.md
├── app.py
├── requirements.txt
└── screenshots/
```

## Getting Started

The complete implementation, data preparation, model training, evaluation, and visualizations are available in the Jupyter Notebook.

The trained model artifacts are not included in this repository at this stage.

## Future Improvements

- Deploy the interactive application.
- Improve classification performance on minority emotion classes.
- Explore additional Transformer architectures.
- Enhance the user interface and prediction explanations.

