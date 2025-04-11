import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained model and tokenizer
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# List of offense labels used by the model
offense_labels = [
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]

def predict_offense(comment):
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.sigmoid(outputs.logits).squeeze().numpy()

    is_offensive = any(score > 0.5 for score in scores)
    offense_types = [label for label, score in zip(offense_labels, scores) if score > 0.5]
    explanation = ", ".join(f"{label}: {score:.2f}" for label, score in zip(offense_labels, scores) if score > 0.5)

    return is_offensive, offense_types, explanation

def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type")

def summarize(df):
    print(f"Loaded {len(df)} comments.")
    print("Sample comments:")
    print(df[['username', 'comment_text']].head())

def generate_plot(df):
    breakdown = df.explode("offense_type")['offense_type'].value_counts()
    breakdown.plot(kind='bar', title='Offense Type Distribution', color='darkred')
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("offense_distribution_open_source.png")
    print("Chart saved as offense_distribution_open_source.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Input CSV or JSON file")
    parser.add_argument("--output", default="flagged_output_open_source.csv", help="Output CSV file")
    args = parser.parse_args()

    df = load_data(args.file)
    summarize(df)

    df["is_offensive"] = False
    df["offense_type"] = [[] for _ in range(len(df))]
    df["explanation"] = ""

    print("\nAnalyzing comments...\n")
    for idx, row in df.iterrows():
        comment = row["comment_text"]
        is_off, types, expl = predict_offense(comment)
        df.at[idx, "is_offensive"] = is_off
        df.at[idx, "offense_type"] = types
        df.at[idx, "explanation"] = expl

    df.to_csv(args.output, index=False)
    print(f"Analysis complete. Results saved to {args.output}")

    print(f"\nTotal offensive comments: {df['is_offensive'].sum()}")
    print("Top 5 offensive comments:")
    print(df[df['is_offensive']].sort_values(by='explanation', ascending=False)[['comment_text', 'offense_type']].head())

    generate_plot(df)

if __name__ == "__main__":
    main()
