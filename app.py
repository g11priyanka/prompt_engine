import streamlit as st
import json
from utils import (
    extract_article,
    get_sentiment,
    detect_category_and_subcategory,
    title_script_generator
)

st.set_page_config(page_title="Web Story Prompt Generator", layout="wide")
st.title("🧠 Generalized Web Story Prompt Engine")

url = st.text_input("Enter a news article URL")

persona = st.selectbox(
    "Choose audience persona:",
    ["genz", "millenial", "working professionals", "creative thinkers", "spiritual explorers"]
)

if url and persona:
    with st.spinner("Analyzing the article and generating prompts..."):
        try:
            # Step 1: Extract article content
            title, summary, full_text = extract_article(url)

            # Step 2: Sentiment analysis
            sentiment = get_sentiment(summary)

            # Step 3: Category, Subcategory, Emotion detection via GPT
            result = detect_category_and_subcategory(full_text)
            category = result["category"]
            subcategory = result["subcategory"]
            emotion = result["emotion"]

            # Step 4: Generate 5-slide prompts using full article context
            output = title_script_generator(
                category, subcategory, emotion, article_text=full_text
            )

            # Merge sentiment, title, summary and persona into the final output
            final_output = {
                "title": title,
                "summary": summary,
                "sentiment": sentiment,
                "emotion": emotion,
                "category": category,
                "subcategory": subcategory,
                "persona": persona,
                "slides": output.get("slides", [])
            }

            st.success("✅ Prompt generation complete!")
            st.json(final_output)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
