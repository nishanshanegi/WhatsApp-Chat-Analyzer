import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from textblob import TextBlob
import helper
import preprocessor

st.markdown("""
    <div style='display: flex; align-items: center; justify-content: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg' width='75' height='75'>
        <h1 style='text-align: center; color: green;'>WhatsApp Chat Analyzer</h1>
    </div>
""", unsafe_allow_html=True)
def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Negative'


st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    user_list = df['user'].unique().tolist()
    user_list = [user for user in user_list if user != 'am']  # Remove 'am' from the list of users
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show analysis"):
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.markdown(
            "<h3 style='text-align: center; font-weight: bold; color: #AF2655;'>Digging into Your Chats: Top Stats Revealed!</h3>",
            unsafe_allow_html=True)

        # Adjust the width parameter to control the spacing between columns
        col1, col2, col3, col4 = st.columns((1, 1, 1, 1))

        with col1:
            st.markdown("<h4 style='text-align: center;'>Messages</h4>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>{}</h2>".format(num_messages), unsafe_allow_html=True)

        with col2:
            st.markdown("<h4 style='text-align: center;'>Words</h4>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>{}</h2>".format(words), unsafe_allow_html=True)

        with col3:
            st.markdown("<h4 style='text-align: center;'>Media Shared</h4>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>{}</h2>".format(num_media_messages), unsafe_allow_html=True)

        with col4:
            st.markdown("<h4 style='text-align: center;'>Links Shared</h4>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>{}</h2>".format(num_links), unsafe_allow_html=True)

        # Sentiment Analysis Section
        st.markdown("<h3 style='text-align: center; font-weight: bold; color: #AF2655;'>Discover the Mood of Your Chats!</h3>", unsafe_allow_html=True)
        df['sentiment'] = df['message'].apply(lambda message: TextBlob(message).sentiment.polarity)
        df['sentiment_label'] = df['sentiment'].apply(categorize_sentiment)
        df = df[df['user'] != 'group_notification']  # Exclude group_notification user from sentiment analysis


        sns.set(style="ticks")
        fig, ax = plt.subplots(figsize=(5, 3))  # Add figsize here to adjust graph size

        sns.countplot(x='sentiment_label', data=df, palette="Set2")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.title("Sentiment Analysis")
        st.pyplot(plt)

        # st.write("Sentiment Analysis by User:")
        # user_sentiment = df.groupby('user')['sentiment'].mean().reset_index()
        # st.bar_chart(user_sentiment.set_index('user'))

        # Sentiment Trends
        # st.title("Sentiment Trends")
        # sentiment_trends = df.groupby(df['only_date'])['sentiment'].mean()
        # fig, ax = plt.subplots(figsize=(5, 3))  # Add figsize here to adjust graph size
        #
        # ax.plot(sentiment_trends.index, sentiment_trends.values, color='blue')
        # plt.xticks(rotation='vertical')
        # st.pyplot(fig)

    # monthly timeline
    st.markdown("<h3 style='text-align: center; font-weight: bold; color: #AF2655;'>Explore Your Chat History Month by Month!</h3>", unsafe_allow_html=True)
    timeline = helper.monthly_timeline(selected_user, df)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(timeline['time'], timeline['message'], color='#D2DE32')
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

    # daily timeline
    st.markdown("<h3 style='text-align: center; font-weight: bold; color: #AF2655;'>Your Chat Habits Every Day</h3>", unsafe_allow_html=True)
    daily_timeline = helper.daily_timeline(selected_user, df)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='#61A3BA')
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

    # activity map
    st.markdown("<h3 style='text-align: center; font-weight: bold;color: #AF2655;'>When Your Chats Were on Overdrive!</h3>", unsafe_allow_html=True)
    col1, col2, = st.columns(2)

    with col1:
        st.header("Most busy day")
        busy_day = helper.week_activity_map(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(busy_day.index, busy_day.values, color='#A0D8B3')
        plt.xticks(rotation='vertical')

        st.pyplot(fig)

    with col2:
        st.header("Most busy month")
        busy_month = helper.month_activity_map(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(busy_month.index, busy_month.values, color='#83764F')
        plt.xticks(rotation='vertical')

        st.pyplot(fig)

    st.markdown("<h3 style='text-align: center; font-weight: bold; color: #AF2655;'>When Do Your Chats Heat Up the Most?</h3>", unsafe_allow_html=True)
    user_heatmap = helper.activity_heatmap(selected_user, df)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(user_heatmap, ax=ax)  # Use the existing ax object

    st.pyplot(fig)

    # Busiest user
    if selected_user == 'Overall':
        st.markdown(
            "<h3 style='text-align: center; font-weight: bold; color: #AF2655;'>Curious Who Sends the Most Messages?</h3>",
            unsafe_allow_html=True)
        x, new_df = helper.most_busy_users(df)

        # Remove 'am' user if present
        if 'am' in x:
            x.drop('am', inplace=True)
            new_df = new_df[new_df['Name'] != 'am']

        fig, ax = plt.subplots()

        col1, col2, = st.columns(2)
        with col1:
            ax.bar(x.index, x.values, color='#95BDFF')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.dataframe(new_df)

    # WordCloud
    st.markdown("<h3 style='text-align: center; font-weight: bold; color: #AF2655;'>What Do Your Words Look Like in a Picture?</h3>", unsafe_allow_html=True)
    df_wc = helper.create_wordcloud(selected_user, df)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.imshow(df_wc)
    st.pyplot(fig)

    # Most common df
    st.markdown("<h3 style='text-align: center; font-weight: bold; color: #AF2655;'>Word Watch: Tracking Our Top Chat Lexicon</h3>", unsafe_allow_html=True)

    most_common_df = helper.most_common_words(selected_user, df)
    fig, ax = plt.subplots()

    ax.barh(most_common_df[0], most_common_df[1], color='#C7D36F')  # Using barh() for a horizontal bar graph
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.title('Most Common Words')
    plt.tight_layout()  # Ensures the labels fit within the figure area

    st.pyplot(fig)