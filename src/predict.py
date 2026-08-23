from transformers import pipeline


# Load a pre-trained AI model for text summarization
summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)


def summarize_text(text):
    """
    Generate an AI-powered summary of the given text.
    """

    result = summarizer(
    "summarize: " + text,
    max_length=60,
    min_length=10,
    do_sample=False,
    num_beams=4,
    no_repeat_ngram_size=3
)

    return result[0]["generated_text"]


def main():
    print("====================================")
    print("   AI Text Summarization Assistant")
    print("====================================")

    while True:
        text = input("\nEnter the text you want to summarize:\n\n")

        if text.strip():
            break

        print("\nPlease enter some text.")

    print("\nGenerating AI summary...")

    summary = summarize_text(text)

    print("\nAI Generated Summary:")
    print("------------------------------------")
    print(summary)
    print("------------------------------------")


if __name__ == "__main__":
    main()