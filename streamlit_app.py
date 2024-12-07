import streamlit as st
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableBranch, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import numpy as np
from collections import Counter
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import requests
from bs4 import BeautifulSoup
import random

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OpenAI_Key"]

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
]

def get_random_user_agent():
    return random.choice(user_agents)

def extract_product_data(url):
    data = {'reviews': [], 'ratings': []}

    try:
        response = requests.get(url, headers={"User-Agent": get_random_user_agent()})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        script_tags = soup.find_all("script", type="application/ld+json")
        for tag in script_tags:
            try:
                data_json = json.loads(tag.string)
                if isinstance(data_json, dict) and data_json.get("@type") == "Product":
                    if "review" in data_json:
                        for review in data_json["review"]:
                            if isinstance(review, dict):
                                review_text = review.get("description", "No Review")
                                rating_value = review.get("reviewRating", {}).get("ratingValue", None)
                                data['reviews'].append(review_text)
                                data['ratings'].append(int(rating_value) if rating_value else None)
                elif isinstance(data_json, dict) and data_json.get("@type") == "Review":
                    review_text = data_json.get("description", "No Review")
                    rating_value = data_json.get("reviewRating", {}).get("ratingValue", None)
                    data['reviews'].append(review_text)
                    data['ratings'].append(int(rating_value) if rating_value else None)
            except (json.JSONDecodeError, AttributeError):
                continue

        return data

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch URL {url}: {e}")
        return None

class ReviewAnalyzer:
    def __init__(self):
        self.client = ChatOpenAI(openai_api_key=st.secrets["OpenAI_Key"], model_name="gpt-3.5-turbo")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    def analyze_reviews(self, data):
        aligned_data = self.align_reviews_and_ratings(data)
        self.handle_overall_reviews(aligned_data)
        self.handle_individual_reviews(aligned_data)

    def align_reviews_and_ratings(self, data):
        checked_data = pd.DataFrame(data)
        checked_data['sentiment'] = checked_data['reviews'].apply(lambda x: self.sentiment_analyzer(x)[0]['label'])
        checked_data['adjusted_rating'] = checked_data.apply(self.adjust_rating, axis=1)
        return checked_data

    def adjust_rating(self, row):
        sentiment = row['sentiment']
        rating = row['ratings']

        if sentiment == 'POSITIVE' and rating < 4:
            return 3
        elif sentiment == 'NEGATIVE' and rating > 2:
            return 3
        elif pd.isna(sentiment):
            return rating
        else:
            return rating

    def handle_overall_reviews(self, checked_data):
        avg_rating = np.mean(checked_data['adjusted_rating'])
        st.write(f"Overall Average Rating: {avg_rating:.2f}")

        all_reviews = " ".join(checked_data['reviews'])
        summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the following product reviews in about 100 words. "
            "If the average rating is less than 4 out of 5, include suggestions to improve. "
            "If the average rating is 4 or higher, identify what to continue keeping.\n\n"
            "Average Rating: {rating}\n"
            "Reviews: {reviews}\n\n"
            "Summary:"
        )
        summary_chain = summary_prompt | self.client
        summary = summary_chain.invoke({"rating": avg_rating, "reviews": all_reviews})

        st.write(f"Overall Summary of Reviews:")
        st.write(summary.content.strip())

    def handle_individual_reviews(self, checked_data):
        positive_reviews = checked_data[checked_data['sentiment'] == 'POSITIVE']['reviews'].tolist()
        negative_reviews = checked_data[checked_data['sentiment'] == 'NEGATIVE']['reviews'].tolist()

        st.write(f"Number of Positive Reviews: {len(positive_reviews)}")
        st.write(f"Number of Negative Reviews: {len(negative_reviews)}")

        fig, ax = plt.subplots()
        ax.bar(['Positive', 'Negative'], [len(positive_reviews), len(negative_reviews)])
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)

        self.analyze_review_group(positive_reviews, "Positive")
        self.analyze_review_group(negative_reviews, "Negative")

    def analyze_review_group(self, reviews, group_name):
        if not reviews:
            st.write(f"No {group_name} reviews to analyze.")
            return

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(reviews))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud of {group_name} Reviews')
        st.pyplot(fig)

        group_reviews = " ".join(reviews)
        summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the following {group_name} product reviews in about 100 words. "
            "{improvement_instruction}\n\n"
            "Reviews: {reviews}\n\n"
            "Summary:"
        )
        improvement_instruction = "Include suggestions for improvement." if group_name == "Negative" else "Highlight key aspects customers appreciate."
        summary_chain = summary_prompt | self.client
        summary = summary_chain.invoke({"group_name": group_name, "improvement_instruction": improvement_instruction, "reviews": group_reviews})

        st.write(f"\nSummary of {group_name} Reviews:")
        st.write(summary.content)

def main():
    st.title("Sentiment Analyzer")

    url = st.text_input("Enter the product webpage URL:")
    if st.button("Analyze Reviews"):
        data = extract_product_data(url)

        if data:
            analyzer = ReviewAnalyzer()
            analyzer.analyze_reviews(data)

if __name__ == "__main__":
    main()
