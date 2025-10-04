import streamlit as st
import requests
import json
import time
import random
from typing import Dict, Any, Tuple

# --- Configuration & Constants ---
# NOTE: The actual API key is provided at runtime by the environment. 
# We use this global variable placeholder as instructed.
API_KEY = "" # The Canvas environment will inject the key if needed.
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={API_KEY}"

# Crisis Resources (Vietnamese and Singaporean, as requested)
CRISIS_MESSAGE = (
    "I hear you, and it sounds like you are going through a lot. You are not alone. "
    "I am an AI and cannot provide professional medical support, but trained help is available right now."
)
HELPLINES = [
    {"name": "Vietnam Youth Helpline", "number": "1900 6233"},
    {"name": "Samaritans of Singapore (SOS)", "number": "+65 1767"}
]

# --- Core LLM API Logic ---

def _make_api_call_with_backoff(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Handles API call with exponential backoff."""
    headers = {'Content-Type': 'application/json'}
    max_retries = 5
    status_code = -1

    for attempt in range(max_retries):
        try:
            # Use requests for the API call in the Python environment
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=15)
            status_code = response.status_code
            
            if status_code == 200:
                return response.json(), status_code
            elif status_code == 429:
                # Handle rate limiting with exponential backoff
                wait_time = 2 ** attempt + random.uniform(0, 1)
                print(f"Rate limit hit (429). Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                # Handle other client/server errors
                print(f"API Error (Status: {status_code}): {response.text}")
                return {"error": f"API call failed with status code {status_code}"}, status_code

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying...")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                return {"error": f"Request failed after {max_retries} attempts."}, -1

    return {"error": "Maximum retries reached."}, status_code


def analyze_sentiment(text: str) -> str:
    """
    A simplified sentiment analysis function.
    In a real application, this would call a dedicated Hugging Face or
    fine-tuned model to classify the emotion (e.g., sadness, stress, anxiety, crisis).
    
    For this prototype, we use keywords to demonstrate the logic.
    """
    text_lower = text.lower()
    
    # Crisis keywords (Triggering the immediate helpline response)
    crisis_keywords = ["give up", "end it", "hopeless", "can't cope anymore", "want to die", "kill myself", "suicide"]
    if any(keyword in text_lower for keyword in crisis_keywords):
        return 'Crisis'

    # Negative/Stress keywords (Triggering modulated, gentle response)
    negative_keywords = ["stressed", "anxious", "sad", "unhappy", "terrible", "bad", "tired of", "struggle", "pressure"]
    if any(keyword in text_lower for keyword in negative_keywords):
        return 'Negative'

    # Neutral/Positive
    return 'Neutral'

def get_ai_response(prompt: str, sentiment: str) -> str:
    """
    Calls the Gemini API to get an empathetic, non-medical response.
    The system instruction is crucial for setting the persona.
    """
    # 1. Define the system instruction for the AI's persona
    system_prompt = (
        "You are 'Tâm Hồn' (Soul/Mind), a friendly, supportive, and empathetic AI companion designed to help high school students "
        "process their feelings. Your role is purely supportive, *never* diagnostic or medical. "
        "Respond in the same language as the user's prompt (Vietnamese or English). "
        "Keep your responses concise, warm, and encouraging. Focus on validating their feelings "
        "and offering simple, healthy coping mechanisms (like taking a break or talking to a trusted adult). "
        "Crucially: NEVER offer medical advice, and always maintain an ethical, non-judgmental tone."
    )
    
    # 2. Adjust the prompt based on detected sentiment
    if sentiment == 'Negative':
        # Add internal guidance to the model for a gentler tone
        augmented_prompt = f"The user is currently expressing feelings of stress/sadness. Respond with extreme gentleness and validation: {prompt}"
    else:
        augmented_prompt = prompt

    # 3. Construct the API payload
    payload = {
        "contents": [{"parts": [{"text": augmented_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # 4. Make the call
    response_json, status = _make_api_call_with_backoff(payload)

    if status == 200 and response_json.get('candidates'):
        text = response_json['candidates'][0]['content']['parts'][0]['text']
        return text
    
    # Fallback for errors
    print(f"AI response error or failed API call: {response_json}")
    return "I'm having trouble connecting right now, but please know I'm listening. Try again in a moment!"

# --- Streamlit Application UI ---

# Set a wide page configuration
st.set_page_config(
    page_title="Tâm Hồn: AI Mental Health Chatbot", 
    layout="wide"
)

st.title("🌱 Tâm Hồn (The Supportive AI) ")

# Ethical Disclaimer box
st.markdown(
    """
    <div style="padding: 10px; border-radius: 8px; background-color: #ffe0e0; border: 1px solid #ff4d4d; margin-bottom: 20px;">
        <p style="font-weight: bold; color: #cc0000;">🚨 Important Disclaimer:</p>
        <p style="color: #cc0000; font-size: 0.9em;">
        Tâm Hồn is an AI support companion, not a substitute for professional medical or psychological advice. 
        If you are in crisis or need professional help, please reach out to the resources listed below.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chào bạn! I'm here to listen. Tell me what's on your mind today, in English or Vietnamese."}
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("I'm listening..."):
    # 1. Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Analyze Sentiment
    sentiment = analyze_sentiment(prompt)

    # 3. Check for Crisis Mode (Highest Priority)
    if sentiment == 'Crisis':
        st.session_state.messages.append({"role": "assistant", "content": CRISIS_MESSAGE})
        with st.chat_message("assistant"):
            st.markdown(CRISIS_MESSAGE)
            
            # Display helpline resources prominently
            st.markdown("---")
            st.markdown("**Urgent Support Lines:**")
            for line in HELPLINES:
                st.markdown(f"- **{line['name']}**: {line['number']}")
            st.markdown("---")
            
    else:
        # 4. Get AI Response (with loading indicator)
        with st.chat_message("assistant"):
            with st.spinner("Tâm Hồn is thinking..."):
                response = get_ai_response(prompt, sentiment)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# --- Final Footer & Information ---
st.sidebar.header("Project Information")
st.sidebar.markdown(f"""
    **Project Phase:** Prototype (v0.9)  
    **Core AI Model:** {GEMINI_MODEL_NAME}  
    **Features Included:** Chat, Sentiment Analysis, Crisis Detection (Vietnamese/English resources).  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Helpline Resources**")
for line in HELPLINES:
    st.sidebar.markdown(f"**{line['name']}**: {line['number']}")

# Code for running the app is handled by 'streamlit run chatbot_app.py'
