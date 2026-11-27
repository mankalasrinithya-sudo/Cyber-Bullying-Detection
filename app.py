
import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
import re
from datetime import datetime


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="CyberShield AI",
    page_icon="🛡️",
    layout="wide"
)


# -------------------------------
# LOAD ALL MACHINE LEARNING MODELS
# -------------------------------

MODEL_PATH = "models/all_models.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Models not found. Please run train_model.py first.")
    st.stop()

model_data = joblib.load(MODEL_PATH)

vectorizer = model_data["vectorizer"]
models = model_data["models"]


# -------------------------------
# TEXT CLEANING
# -------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -------------------------------
# DATABASE SETUP
# -------------------------------

os.makedirs("database", exist_ok=True)


def create_database():

    conn = sqlite3.connect("database/history.db")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            prediction TEXT,
            score REAL,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_history(message, prediction):

    conn = sqlite3.connect("database/history.db")

    conn.execute("""
        INSERT INTO history (message, prediction, score, date)
        VALUES (?, ?, ?, ?)
    """, (
        message,
        prediction,
        None,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_history():

    conn = sqlite3.connect("database/history.db")

    history = pd.read_sql_query(
        "SELECT * FROM history ORDER BY id DESC",
        conn
    )

    conn.close()

    return history


create_database()


# -------------------------------
# CUSTOM DESIGN
# -------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fc;
}

[data-testid="stSidebar"] {
    background-color: #172554;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #172554;
}

.subtitle {
    color: #64748b;
    font-size: 16px;
}

.result-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------

st.sidebar.title("🛡️ CyberShield AI")
st.sidebar.caption("Cyberbullying Detection System")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Home", "Detect Cyberbullying", "History", "About"]
)

st.sidebar.divider()
st.sidebar.caption("AI-powered text analysis")


# -------------------------------
# HOME PAGE
# -------------------------------

if page == "Home":

    history = get_history()

    total = len(history)

    bullying = len(
        history[history["prediction"] == "Cyberbullying"]
    )

    safe = len(
        history[history["prediction"] == "Not Cyberbullying"]
    )

    st.markdown(
        '<div class="main-title">Welcome to CyberShield AI 🛡️</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">An intelligent system for detecting '
        'cyberbullying in online messages.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    col1.metric("Messages Analysed", total)
    col2.metric("Potentially Harmful", bullying)
    col3.metric("Non-Bullying", safe)

    st.write("")

    st.subheader("About the System")

    st.write(
        "CyberShield AI uses Natural Language Processing and "
        "Machine Learning to analyse text messages and identify "
        "potentially harmful content."
    )

    st.info(
        "Enter a social media comment in the detection page "
        "to analyse it."
    )

    if total > 0:

        st.subheader("Detection Statistics")

        chart_data = pd.DataFrame({
            "Category": ["Cyberbullying", "Not Cyberbullying"],
            "Messages": [bullying, safe]
        })

        st.bar_chart(
            chart_data.set_index("Category")
        )


# -------------------------------
# DETECTION PAGE
# -------------------------------

elif page == "Detect Cyberbullying":

    st.markdown(
        '<div class="main-title">Message Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter a message below to check for potentially harmful content."
    )

    with st.form("detection_form"):

        # Select one of the three trained models
        selected_model = st.selectbox(
            "Select Machine Learning Model",
            [
                "Logistic Regression",
                "Linear SVM",
                "Naive Bayes"
            ]
        )

        message = st.text_area(
            "Enter your message",
            placeholder="Type or paste a social media comment here...",
            height=150
        )

        submitted = st.form_submit_button(
            "🔍 Analyse Message",
            use_container_width=True
        )

    if submitted:

        if not message.strip():

            st.warning("Please enter a message first.")

        else:

            # Step 1: Clean the message
            cleaned_message = clean_text(message)

            # Step 2: Convert text into TF-IDF features
            message_tfidf = vectorizer.transform([cleaned_message])

            # Step 3: Select the chosen trained model
            model = models[selected_model]

            # Step 4: Predict the category
            prediction = model.predict(message_tfidf)[0]

            # Step 5: Convert numeric prediction into readable text
            if prediction == 1:
                result = "Cyberbullying"
            else:
                result = "Not Cyberbullying"

            # Step 6: Save the result in database
            save_history(message, result)

            # Step 7: Display the result
            st.divider()
            st.subheader("Analysis Result")

            st.write("Selected Model:", selected_model)

            if prediction == 1:

                st.error(
                    "⚠️ Potential Cyberbullying Detected"
                )

                st.write(
                    "The model identifies this message as potentially "
                    "harmful based on patterns learned from its training data."
                )

            else:

                st.success(
                    "✅ No Cyberbullying Detected"
                )

                st.write(
                    "The model identifies this message as non-cyberbullying "
                    "based on patterns learned from its training data."
                )

            st.info(
                "Automated text analysis may make mistakes. "
                "Consider the message context before making decisions."
            )


# -------------------------------
# HISTORY PAGE
# -------------------------------

elif page == "History":

    st.markdown(
        '<div class="main-title">Detection History</div>',
        unsafe_allow_html=True
    )

    history = get_history()

    if history.empty:

        st.info("No messages have been analysed yet.")

    else:

        st.write("Previously analysed messages:")

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear History"):

            conn = sqlite3.connect("database/history.db")

            conn.execute("DELETE FROM history")

            conn.commit()
            conn.close()

            st.success("History cleared successfully!")

            st.rerun()


# -------------------------------
# ABOUT PAGE
# -------------------------------

elif page == "About":

    st.markdown(
        '<div class="main-title">About CyberShield AI</div>',
        unsafe_allow_html=True
    )

    st.write("""
    CyberShield AI is a Python-based machine learning application
    designed to identify potentially harmful messages.

    The project uses Natural Language Processing and supervised
    machine learning techniques.

    **Technologies Used**

    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - TF-IDF Vectorization
    - Logistic Regression
    - Linear Support Vector Machine (SVM)
    - Multinomial Naive Bayes
    - SQLite

    **How It Works**

    1. The user enters a text message.
    2. The message is cleaned using text preprocessing.
    3. TF-IDF converts the text into numerical features.
    4. The selected machine learning model predicts the category.
    5. The result is displayed and saved in the database.

    **Disclaimer**

    This application is an academic demonstration.
    It cannot understand every social situation, sarcasm,
    or conversation context. Its predictions should not
    be treated as definitive judgements.
    """)

    st.divider()

    st.caption(
        "CyberShield AI | Python Mini Project"
    )
