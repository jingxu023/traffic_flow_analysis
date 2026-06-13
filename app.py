import os
import logging
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.prediction import build_features, load_model_bundle, predict_traffic

# =========================
# OpenAI setup
# =========================
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# =========================
# Page settings
# =========================
st.set_page_config(
    page_title="Traffic Analytics Dashboard",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Traffic Analytics Dashboard")
st.write("Predict traffic conditions and explore mobility insights.")

# =========================
# Load model and encoders
# =========================
@st.cache_resource
def load_model():
    return load_model_bundle()


model_bundle = load_model()

# =========================
# Chat history
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# User input
# =========================
st.subheader("Traffic Input")

col1, col2 = st.columns(2)

with col1:
    time = st.slider("Time of day", 0, 23, 8)
    day = st.selectbox("Day of the week", model_bundle.day_encoder.classes_)

with col2:
    car_count = st.number_input("Car Count", min_value=0, value=50)
    bike_count = st.number_input("Bike Count", min_value=0, value=10)

bus_count = st.number_input("Bus Count", min_value=0, value=5)
truck_count = st.number_input("Truck Count", min_value=0, value=10)

total = car_count + bike_count + bus_count + truck_count

st.write(f"**Total Vehicles:** {total}")

# =========================
# Prediction
# =========================
input_data = build_features(
    model_bundle,
    hour=time,
    day=day,
    car_count=car_count,
    bike_count=bike_count,
    bus_count=bus_count,
    truck_count=truck_count,
)

if st.button("Predict Traffic Situation"):
    prediction_label = predict_traffic(model_bundle, input_data)

    st.success(f"Predicted Traffic Situation: **{prediction_label}**")

    st.session_state.last_prediction = {
        "time": time,
        "day": day,
        "car_count": car_count,
        "bike_count": bike_count,
        "bus_count": bus_count,
        "truck_count": truck_count,
        "total": total,
        "prediction": prediction_label
    }

# =========================
# AI Traffic Analyst
# =========================
st.subheader("🤖 AI Traffic Analyst Assistant")

def build_prediction_context():
    if "last_prediction" not in st.session_state:
        return """
        No prediction has been made yet.
        The user can enter traffic data and predict the traffic situation.
        """

    p = st.session_state.last_prediction

    return f"""
    Traffic prediction context:
    - Day: {p["day"]}
    - Time: {p["time"]}:00
    - Car count: {p["car_count"]}
    - Bike count: {p["bike_count"]}
    - Bus count: {p["bus_count"]}
    - Truck count: {p["truck_count"]}
    - Total vehicles: {p["total"]}
    - Predicted traffic situation: {p["prediction"]}
    """


def ask_ai_traffic_analyst(question, context, chat_history):
    if client is None:
        return "The AI analyst is disabled. Set OPENAI_API_KEY to enable it."

    history_text = ""

    for msg in chat_history[-6:]:
        history_text += f'{msg["role"]}: {msg["content"]}\n'

    prompt = f"""
    You are an AI traffic data analyst working for a mobility technology team.

    Use the traffic prediction context below to answer the user's question.
    If there is not enough information, say that clearly.

    Traffic context:
    {context}

    Recent conversation:
    {history_text}

    User question:
    {question}

    Answer clearly and professionally.
    Include practical mobility or traffic-management insights when relevant.
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text


example_questions = [
    "What does this traffic situation mean?",
    "Which vehicle type contributes most to traffic volume?",
    "What could mobility planners learn from this result?",
    "Give one recommendation to reduce congestion."
]

selected_question = st.selectbox(
    "Choose an example question:",
    example_questions
)

custom_question = st.text_input("Or ask your own question:")

final_question = custom_question if custom_question else selected_question

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.button("Ask AI Traffic Analyst"):
    context = build_prediction_context()

    st.session_state.messages.append({
        "role": "user",
        "content": final_question
    })

    with st.chat_message("user"):
        st.write(final_question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing traffic situation..."):
            answer = ask_ai_traffic_analyst(
                final_question,
                context,
                st.session_state.messages
            )
            st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
