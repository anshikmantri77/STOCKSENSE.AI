import streamlit as st


@st.cache_resource
def load_finbert():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        return_all_scores=True,
    )


def get_sentiment(headlines: list[str]) -> dict:
    if not headlines:
        return {"positive": 0, "negative": 0, "neutral": 1}

    finbert = load_finbert()
    results = finbert(headlines, batch_size=8, truncation=True, max_length=512)

    scores = {"positive": 0, "negative": 0, "neutral": 0}
    for r in results:
        for label_score in r:
            label = label_score["label"].lower()
            scores[label] += label_score["score"]

    total = len(results)
    if total > 0:
        return {k: round(v / total, 3) for k, v in scores.items()}
    return {"positive": 0, "negative": 0, "neutral": 1}
