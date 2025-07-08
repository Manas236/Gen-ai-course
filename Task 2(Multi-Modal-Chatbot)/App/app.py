import os
import json
import base64
import time
import requests
import subprocess
import streamlit as st
from PIL import Image
from io import BytesIO
import joblib
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

vectorizer = joblib.load("../Model/tfidf_vectorizer.pkl")
intent_model = joblib.load("../Model/chatbot_intent_classifier.pkl")

def predict_intent(text):
    X = vectorizer.transform([text])
    prediction = intent_model.predict(X)
    return prediction[0]

def to_jpeg_bytes(image):
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()

def gemini_text_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini text error: {e}"

def gemini_image_response(prompt, image_bytes):
    """
    Pass the image **first** and the prompt **second** – this is required
    by Gemini 1.5’s multimodal interface.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [{"mime_type": "image/jpeg", "data": image_bytes}, prompt]
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini image error: {e}"

def llava_response_api(prompt, image_bytes):
    b64_img = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [b64_img],
                "stream": True
            },
            stream=True
        )
        reply = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.strip():
                try:
                    data = json.loads(line)
                    reply += data.get("response", "")
                except json.JSONDecodeError:
                    print("⚠️ Invalid JSON chunk:", line)
        return reply.strip()
    except Exception as e:
        return f"❌ LLaVA API error: {e}"

def llama_chat_response(prompt):
    cmd = ["ollama", "run", "llama3", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else f"❌ LLaMA3 error:\n{result.stderr}"

def generate_image(prompt):
    try:
        response = requests.post(
            url="http://127.0.0.1:7860/sdapi/v1/txt2img",
            json={"prompt": prompt}
        )
        if response.status_code == 200:
            r = response.json()
            image_data = r["images"][0]
            image_bytes = base64.b64decode(image_data)
            return Image.open(BytesIO(image_bytes))
        else:
            st.error(f"SD Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ SD error: {e}")
        return None

if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_image_bytes" not in st.session_state:
    st.session_state.last_image_bytes = None
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Use LLaMA3 & LLaVA"

st.set_page_config(page_title="🧠 Local Multi-Modal Chatbot", layout="centered")
st.title("🧠 Multi-Modal Chatbot (Gemini | LLaVA | LLaMA | SD | Intent AI)")

st.session_state.model_choice = st.radio("Model Mode", ["Use Gemini", "Use LLaMA3 & LLaVA"], horizontal=True)

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.write(message["content"])
        elif message["type"] == "image":
            st.image(message["content"])

col1, col2 = st.columns([5, 2])
with col1:
    prompt = st.chat_input("Type your prompt...")
with col2:
    uploaded_file = st.file_uploader("📎", type=["jpg", "jpeg", "png"], key="chat_attach", label_visibility="collapsed")

if prompt:
    st.session_state.chat.append({"role": "user", "type": "text", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    image_bytes = None
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_bytes = to_jpeg_bytes(image)
        st.image(image, caption="Attached Image")
        st.session_state.chat.append({"role": "user", "type": "image", "content": image, "image_bytes": image_bytes})
        st.session_state.last_image_bytes = image_bytes

    intent = predict_intent(prompt.lower().strip())
    if intent not in ["image_description", "image_generation"] and st.session_state.last_image_bytes:
        intent = "image_description"

    if st.session_state.model_choice == "Use Gemini":

        if intent == "image_description" and st.session_state.last_image_bytes:
            with st.chat_message("assistant"):
                st.write("Analyzing image with Gemini...")
                reply = gemini_image_response(prompt, st.session_state.last_image_bytes)
                st.write(reply)
                st.session_state.chat.append({"role": "assistant", "type": "text", "content": reply})

        elif intent == "image_generation":
            with st.chat_message("assistant"):
                st.write("Generating image using Stable Diffusion...")
                generated_img = generate_image(prompt)
                if generated_img:
                    st.image(generated_img, caption="Generated Image")
                    st.session_state.last_image_bytes = to_jpeg_bytes(generated_img)
                    st.session_state.chat.append({"role": "assistant", "type": "image", "content": generated_img})
                else:
                    st.write("❌ Failed to generate image.")

        else:
            with st.chat_message("assistant"):
                st.write("Thinking with Gemini...")
                reply = gemini_text_response(prompt)
                st.write(reply)
                st.session_state.chat.append({"role": "assistant", "type": "text", "content": reply})

    else:
        if intent == "image_description" and st.session_state.last_image_bytes:
            with st.chat_message("assistant"):
                st.write("Analyzing image with LLaVA...")
                reply = llava_response_api(prompt, st.session_state.last_image_bytes)
                st.write(reply)
                st.session_state.chat.append({"role": "assistant", "type": "text", "content": reply})

        elif intent == "image_generation":
            with st.chat_message("assistant"):
                st.write("Generating image using Stable Diffusion...")
                generated_img = generate_image(prompt)
                if generated_img:
                    st.image(generated_img, caption="Generated Image")
                    st.session_state.last_image_bytes = to_jpeg_bytes(generated_img)
                    st.session_state.chat.append({"role": "assistant", "type": "image", "content": generated_img})
                else:
                    st.write("❌ Failed to generate image.")
        else:
            with st.chat_message("assistant"):
                st.write("Thinking with LLaMA3...")
                reply = llama_chat_response(prompt)
                st.write(reply)
                st.session_state.chat.append({"role": "assistant", "type": "text", "content": reply})

    if "chat_attach" in st.session_state:
        del st.session_state["chat_attach"]
        st.toast("File processed, refreshing…")
        time.sleep(2)
        st.rerun()
