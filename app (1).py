import os
import torch
import gradio as gr
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "emotion_classification_artifacts/DistilBERT"
)
MAX_LENGTH = 128
TARGET_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

EMOTION_MAPPING = {
    "admiration": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "joy": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "sadness": "sadness",
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "fear": "fear",
    "nervousness": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Fine-tuned model folder not found: {MODEL_DIR}\n"
        "Place the saved DistilBERT model folder at this location."
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()


# ---------------------------------------------------------
# Load and prepare real test-set examples
# ---------------------------------------------------------
def prepare_test_examples():
    dataset = load_dataset(
        "google-research-datasets/go_emotions",
        "simplified"
    )

    label_names = dataset["test"].features["labels"].feature.names
    examples = []

    for index, row in enumerate(dataset["test"]):
        mapped = {
            EMOTION_MAPPING[label_names[label_id]]
            for label_id in row["labels"]
            if label_names[label_id] in EMOTION_MAPPING
        }

        # Keep only examples with exactly one mapped target emotion.
        if len(mapped) == 1:
            examples.append({
                "index": index,
                "text": row["text"]
            })

    return examples


test_examples = prepare_test_examples()
sentence_choices = [
    f"Test example {item['index']} | {item['text']}"
    for item in test_examples
]


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
def predict_emotion(text):
    if not text or not text.strip():
        return "Please select or enter a sentence.", {}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    scores = {
        emotion: float(probabilities[index].item())
        for index, emotion in enumerate(TARGET_EMOTIONS)
    }
    predicted = max(scores, key=scores.get)

    return predicted.upper(), scores


def analyze_dataset_sentence(selected):
    if not selected:
        return "Please select a sentence first.", {}

    text = selected.split(" | ", 1)[1]
    return predict_emotion(text)


# ---------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------
CSS = """
.gradio-container {
    max-width: 1050px !important;
    margin: auto !important;
}
#main-title { text-align: center; }
#subtitle {
    text-align: center;
    color: #64748b;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=CSS,
    title="Emotion Classification Studio"
) as demo:

    gr.Markdown(
        """
        # Emotion Classification Studio
        <div id="subtitle">
        AI-powered text emotion detection using a fine-tuned DistilBERT model
        </div>
        """
    )

    gr.Markdown("## Explore and Analyze Emotions")

    input_mode = gr.Radio(
        choices=["Dataset Examples", "Custom Text"],
        value="Dataset Examples",
        label="Choose Input Method"
    )

    with gr.Group(visible=True) as dataset_section:
        sentence_choice = gr.Dropdown(
            choices=sentence_choices,
            label="Choose a sentence from the GoEmotions test dataset",
            value=None,
            interactive=True,
            filterable=True
        )
        dataset_analyze = gr.Button(
            "Analyze Sentence",
            variant="primary"
        )

    with gr.Group(visible=False) as custom_section:
        custom_input = gr.Textbox(
            label="Enter your text",
            placeholder="Type a sentence to analyze its emotion...",
            lines=4
        )
        custom_analyze = gr.Button(
            "Analyze Text",
            variant="primary"
        )

    prediction_output = gr.Textbox(
        label="Predicted Emotion",
        interactive=False
    )

    confidence_output = gr.Label(
        label="Confidence Scores — All 6 Emotions",
        num_top_classes=6
    )

    def switch_mode(mode):
        return (
            gr.update(visible=(mode == "Dataset Examples")),
            gr.update(visible=(mode == "Custom Text"))
        )

    input_mode.change(
        fn=switch_mode,
        inputs=input_mode,
        outputs=[dataset_section, custom_section]
    )

    dataset_analyze.click(
        fn=analyze_dataset_sentence,
        inputs=sentence_choice,
        outputs=[prediction_output, confidence_output]
    )

    custom_analyze.click(
        fn=predict_emotion,
        inputs=custom_input,
        outputs=[prediction_output, confidence_output]
    )

    gr.Markdown(
        """
        ---
        **Model:** DistilBERT  
        **Dataset:** GoEmotions  
        **Classes:** Joy · Sadness · Anger · Fear · Surprise · Disgust
        """
    )

if __name__ == "__main__":
    demo.launch()
