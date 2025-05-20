# hotstar_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

st.set_page_config(layout="wide")
sns.set(style="whitegrid")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_hotstar_data.csv")
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df.fillna('Unknown', inplace=True)
    return df

df = load_data()

# Sidebar options
st.sidebar.title("📺 JioHotstar Analytics")
option = st.sidebar.radio("Choose Analysis", ["📈 Content Trends", "🎭 Similar Titles", "🔗 Collaboration Network", "🗣️ Sentiment Analysis"])

# --------------------------------------
if option == "📈 Content Trends":
    st.title("📈 Content Trend Analysis")

    trend = df.groupby(['year_added', 'type']).size().unstack().fillna(0)
    st.bar_chart(trend)

    st.subheader("Year-wise Release Count")
    release = df['release_year'].value_counts().sort_index()
    fig, ax = plt.subplots()
    release.plot(ax=ax, marker='o', color='green')
    plt.grid()
    st.pyplot(fig)

# --------------------------------------
elif option == "🎭 Similar Titles":
    st.title("🎭 Content Similarity Recommendation")

    title_input = st.text_input("Enter a title to find similar ones:")
    if title_input:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['description'].fillna(''))
        cosine_sim = cosine_similarity(tfidf_matrix)

        idx = df[df['title'].str.lower() == title_input.lower()].index
        if not idx.empty:
            idx = idx[0]
            scores = list(enumerate(cosine_sim[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
            st.subheader("Top 5 Similar Titles:")
            for i, score in scores:
                st.write(f"{df.iloc[i]['title']} — Similarity Score: {score:.2f}")
        else:
            st.warning("Title not found!")

# --------------------------------------
elif option == "🔗 Collaboration Network":
    st.title("🔗 Actor-Director Collaboration Network")

    top_directors = df[df['director'] != "Unknown"]['director'].value_counts().head(10).index.tolist()
    director_select = st.selectbox("Select a Director", top_directors)

    if director_select:
        G = nx.Graph()
        for _, row in df[df['director'] == director_select].iterrows():
            actors = row['cast'].split(", ") if row['cast'] != "Unknown" else []
            for actor in actors:
                G.add_edge(director_select, actor.strip())

        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
                node_size=600, font_size=8)
        st.pyplot(fig)

# --------------------------------------
elif option == "🗣️ Sentiment Analysis":
    st.title("🗣️ Audience Review Sentiment")

    reviews = st.text_area("Paste reviews (one per line)", height=150)

    if reviews:
        review_list = reviews.strip().split("\n")
        sentiments = [TextBlob(review).sentiment.polarity for review in review_list]

        sentiment_df = pd.DataFrame({
            'Review': review_list,
            'Polarity': sentiments
        })
        sentiment_df['Sentiment'] = sentiment_df['Polarity'].apply(
            lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

        st.write(sentiment_df)

        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=sentiment_df, x='Sentiment', palette='coolwarm', ax=ax)
        st.pyplot(fig)
