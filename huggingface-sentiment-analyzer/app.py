import gradio as gr

from src.sentiment import analyze_sentiment


def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text.", "—"

    label, score = analyze_sentiment(text)

    return label, f"{score:.2%}"


with gr.Blocks(title="Hugging Face Sentiment Analyzer") as demo:

    gr.Markdown(
        """
        # 🤖 Hugging Face Sentiment Analyzer

        Analyze the sentiment of any text using a
        **pre-trained DistilBERT model from Hugging Face**.
        """
    )

    with gr.Row():

        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your text",
                placeholder="Example: I really enjoyed this movie!",
                lines=8
            )

            with gr.Row():
                analyze_button = gr.Button(
                    "🔍 Analyze Sentiment",
                    variant="primary"
                )

                clear_button = gr.ClearButton(
                    components=[text_input]
                )

        with gr.Column():
            sentiment_output = gr.Textbox(
                label="Sentiment"
            )

            confidence_output = gr.Textbox(
                label="Confidence"
            )

    gr.Examples(
        examples=[
            ["I absolutely loved this product!"],
            ["This movie was boring and disappointing."],
            ["The experience was amazing and enjoyable!"],
            ["The service was terrible."]
        ],
        inputs=text_input
    )

    analyze_button.click(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[
            sentiment_output,
            confidence_output
        ]
    )


demo.launch()