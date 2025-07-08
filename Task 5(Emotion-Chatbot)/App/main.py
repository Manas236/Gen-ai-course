import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import google.generativeai as genai
import os
import dotenv
import google.generativeai as genai


dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
analyzer = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Hybrid Sentiment Chatbot", layout="centered")
st.title("🧠 Sentiment-Aware Chatbot (VADER + Gemini)")
st.write("Talk to the bot — it understands your emotions and responds smartly!")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def categorize_sentiment(compound):
    if compound >= 0.6:
        return "very positive"
    elif 0.2 <= compound < 0.6:
        return "positive"
    elif -0.2 < compound < 0.2:
        return "neutral"
    elif -0.6 <= compound <= -0.2:
        return "negative"
    else:
        return "very negative"

def get_gemini_response(user_input, sentiment_label):
    prompt = f"""
    You are an emotionally intelligent chatbot. The user is feeling {sentiment_label}.
    Reply empathetically to this message, offering help or engagement:

    User: "{user_input}"

    Your reply:
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return "Sorry, I couldn't process that right now."


for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧍 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type something...")
    submit = st.form_submit_button("Send")


if submit and user_input:
    scores = analyzer.polarity_scores(user_input)
    compound = scores['compound']
    sentiment_label = categorize_sentiment(compound)

    bot_response = get_gemini_response(user_input, sentiment_label)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", f"{bot_response} (Sentiment: {sentiment_label}, Score: {compound:.2f})"))
    st.rerun()

