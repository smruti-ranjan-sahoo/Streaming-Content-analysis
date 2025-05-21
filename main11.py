import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import statsmodels.api as sm

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
option = st.sidebar.radio("Choose Analysis", [
    "📈 Content Trends",
    "🎭 Similar Titles",
    "🔗 Collaboration Network",
     "🗣️ Sentiment Analysis"
])

# -------------------------------------------------------------------------------------------------
if option == "📈 Content Trends":
    st.title("📈 Content Trend Analysis with Forecasts")

    # Original trend: yearly count by type
    trend = df.groupby(['year_added', 'type']).size().unstack().fillna(0)
    st.subheader("Yearly Release Counts by Content Type")
    st.bar_chart(trend)

    # Year-wise total release count plot
    st.subheader("Year-wise Total Release Count")
    release = df['release_year'].value_counts().sort_index()
    fig, ax = plt.subplots()
    release.plot(ax=ax, marker='o', color='green')
    plt.grid()
    st.pyplot(fig)

    # Prepare data for forecasting
    data = release.values.reshape(-1, 1)
    years = release.index.values

    # Scaling for CNN-LSTM (using MinMaxScaler)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    seq_len = 5  # sequence length for time series input

    # Create sequences for CNN-LSTM input
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, seq_len)

    # Build and train CNN-LSTM model (cache to avoid retraining every rerun)
    @st.cache_resource(show_spinner=False)
    def build_and_train_cnn_lstm(X, y):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            tf.keras.layers.MaxPooling1D(pool_size=1),
            tf.keras.layers.LSTM(50, activation='tanh'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model.fit(X, y, epochs=200, validation_split=0.1, callbacks=[early_stop], verbose=0)
        return model

    model = build_and_train_cnn_lstm(X, y)

    # Forecast next 5 years using CNN-LSTM
    last_seq = scaled_data[-seq_len:]
    predictions = []
    for _ in range(5):
        pred = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)
        predictions.append(pred[0, 0])
        last_seq = np.vstack([last_seq[1:], pred])

    predicted_counts_cnn_lstm = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

    # Plot CNN-LSTM forecast
    forecast_years = range(years[-1] + 1, years[-1] + 6)
    st.subheader("CNN-LSTM Forecast")
    fig2, ax2 = plt.subplots()
    ax2.plot(years, data.flatten(), label="Original")
    ax2.plot(forecast_years, predicted_counts_cnn_lstm, linestyle='--', marker='o', label="CNN-LSTM Forecast")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Number of Releases")
    ax2.set_title("JioHotstar Forecast (CNN-LSTM)")
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)

    # ARIMA forecast
    ts = pd.Series(data.flatten(), index=pd.Index(years, name='Year'))
    arima_model = sm.tsa.ARIMA(ts, order=(2,1,2)).fit()
    forecast_arima = arima_model.forecast(steps=5)

    # Plot ARIMA forecast
    st.subheader("ARIMA Forecast")
    fig3, ax3 = plt.subplots()
    ax3.plot(ts.index, ts.values, label="Original")
    ax3.plot(range(ts.index[-1] + 1, ts.index[-1] + 6), forecast_arima, linestyle='--', marker='o', color='green', label="ARIMA Forecast")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Number of Releases")
    ax3.set_title("JioHotstar Content Forecast using ARIMA")
    ax3.legend()
    ax3.grid()
    st.pyplot(fig3)

# ---------------------------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------------------------
elif option == "🔗 Collaboration Network":
    st.title("🔗 Actor-Director Collaboration Network")

    top_directors = df[df['director'] != "Unknown"]['director'].value_counts().head(100).index.tolist()
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

# -----------------------------------------------------------------------------------------------
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
