import streamlit as st
from emotion_bert import EmotionDetector
from llm import generate_response

st.set_page_config(page_title="MoodMate", page_icon="🫂")

detector = EmotionDetector()

st.title("🫂 MoodMate")
st.markdown("Your friendly companion.")

user_input = st.text_input("You:", placeholder="Just let it out...")

if user_input:
    emotion, confidence = detector.predict(user_input)
    st.markdown(f"**Detected Emotion:** `{emotion}` ({confidence:.2f})")

    with st.spinner("Generating response..."):
        bot_reply = generate_response(user_input, emotion)

    st.success(bot_reply)
