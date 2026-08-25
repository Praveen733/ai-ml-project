from transformers import pipeline


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# Load the model once when the application starts
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME
)


def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]

    label = result["label"]
    score = result["score"]

    return label, score