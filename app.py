import os
import pickle
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# OpenAI setup
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Page settings
# =========================
st.set_page_config(
    page_title="Traffic Situation Prediction App",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Traffic Situation Prediction App")
st.write("Predict traffic situation and ask the AI Traffic Analyst for insights.")

# =========================
# Load model and encoders
# =========================
@st.cache_resource
def load_model():
    with open("traffic_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("label_encoder_day.pkl", "rb") as file:
        le_day = pickle.load(file)

    with open("label_encoder_target.pkl", "rb") as file:
        le_target = pickle.load(file)

    return model, le_day, le_target


model, le_day, le_target = load_model()

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
    day = st.selectbox("Day of the week", le_day.classes_)

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
encoded_day = le_day.transform([day])[0]

input_data = pd.DataFrame({
    "Date": [time],
    "Day of the week": [encoded_day],
    "CarCount": [car_count],
    "BikeCount": [bike_count],
    "BusCount": [bus_count],
    "TruckCount": [truck_count]
})

if st.button("Predict Traffic Situation"):
    prediction = model.predict(input_data)[0]
    prediction_label = le_target.inverse_transform([prediction])[0]

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