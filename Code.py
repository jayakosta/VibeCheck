import os
from googleapiclient.discovery import build
import pandas as pd
import nltk
import emoji
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

API_KEY = "AIzaSyCarHUCMkutE-Hb1iLekjM1NIomOXiO4OA"
youtube = build("youtube", "v3", developerKey=API_KEY)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
bert_sentiment = pipeline("sentiment-analysis")

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = emoji.demojize(text)
    text = re.sub(r':[a-zA-Z_]+:', '', text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def get_video_metadata(video_id):
    try:
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()

        if "items" in response and response["items"]:
            snippet = response["items"][0]["snippet"]
            return snippet["title"], snippet["description"]
        return "", ""
    except Exception as e:
        print(f"Error fetching video metadata: {e}")
        return "", ""

def get_all_comments(video_id):
    comments = []
    next_page_token = None
    
    try:
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get("items", []):
                comment_data = {
                    'text': item["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
                    'timestamp': item["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
                }
                comments.append(comment_data)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except Exception as e:
        print(f"Error fetching comments: {e}")
    
    return comments

def extract_topics(comments, num_topics=5):
    if not comments:
        return {}

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(comments)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    topic_keywords = {}
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[:-10 - 1:-1]]
        topic_keywords[topic_idx] = top_words
    
    return topic_keywords

def is_comment_relevant(comment, video_keywords):
    return any(keyword in comment.lower() for keyword in video_keywords)

def classify_context_sentiment(comment, video_keywords):
    try:
        sentiment_result = bert_sentiment(comment)[0]
        sentiment = sentiment_result['label']
        
        if sentiment == "NEGATIVE" and not is_comment_relevant(comment, video_keywords):
            return "POSITIVE"
        return sentiment
    except Exception as e:
        print(f"Error in sentiment classification: {e}")
        return "NEUTRAL"

video_id = "_kfCQU1k5WI"
title, description = get_video_metadata(video_id)
video_keywords = set(clean_text(title + " " + description).split())

comments = get_all_comments(video_id)
df = pd.DataFrame(comments)
df.columns = ["Comment", "timestamp"]

df["Cleaned_Comment"] = df["Comment"].apply(clean_text)
df["Context_Sentiment"] = df["Cleaned_Comment"].apply(lambda x: classify_context_sentiment(x, video_keywords))

topic_data = extract_topics(df["Cleaned_Comment"].tolist())
df["Topic"] = df["Cleaned_Comment"].apply(lambda x: next((t for t, words in topic_data.items() if any(word in x for word in words)), "Other"))

sentiment_counts = df["Context_Sentiment"].value_counts(normalize=True) * 100
positive_score = sentiment_counts.get("POSITIVE", 0)
negative_score = sentiment_counts.get("NEGATIVE", 0)
neutral_score = sentiment_counts.get("NEUTRAL", 0)

print("\n🔹 Sentiment Analysis Results:")
print(f"Positive: {positive_score:.2f}%")
print(f"Negative: {negative_score:.2f}%")
print(f"Neutral: {neutral_score:.2f}%")

print("\n🔹 Suggestions Based on Sentiment Analysis:")
if negative_score > 50:
    print("⚠️ Many comments are negative. Consider addressing concerns or clarifying misunderstandings.")
elif neutral_score > 50:
    print("ℹ️ Many comments are neutral. Try engaging with viewers to encourage more discussion.")
elif positive_score > 60:
    print("🎉 Most comments are positive! Keep up the good work and engage with your audience.")
elif negative_score > 30:
    print("⚠️ Some negative feedback is present. Review common complaints and address them accordingly.")
else:
    print("👍 Sentiment is balanced. Keep engaging with your audience.")

df.to_csv("context_aware_sentiment_analysis.csv", index=False)
print("\n✅ Processed data saved as 'context_aware_sentiment_analysis.csv'")
