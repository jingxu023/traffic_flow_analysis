import streamlit as st
import pandas as pd
import pickle


# Page settings
st.set_page_config(
    page_title="Traffic Situation Prediction App",
    page_icon="🚦",
    layout="centered"
)


# Load model and encoders
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


# App title
st.title("🚦 Traffic Situation Prediction App")

st.write(
    "This app predicts the traffic situation based on date, day of the week, "
    "and vehicle counts."
)


# User input section
st.header("Enter Traffic Information")

date = st.number_input(
    "Date",
    min_value=1,
    max_value=31,
    value=1,
    step=1
)

day = st.selectbox(
    "Day of the Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

car_count = st.number_input(
    "Car Count",
    min_value=0,
    value=50,
    step=1
)

bike_count = st.number_input(
    "Bike Count",
    min_value=0,
    value=10,
    step=1
)

bus_count = st.number_input(
    "Bus Count",
    min_value=0,
    value=10,
    step=1
)

truck_count = st.number_input(
    "Truck Count",
    min_value=0,
    value=10,
    step=1
)


# Prediction
if st.button("Predict Traffic Situation"):
    day_encoded = le_day.transform([day])[0]

    input_data = pd.DataFrame(
        [[date, day_encoded, car_count, bike_count, bus_count, truck_count]],
        columns=["Date", "Day of the week", "CarCount", "BikeCount", "BusCount", "TruckCount"]
    )

    prediction_encoded = model.predict(input_data)[0]
    prediction = le_target.inverse_transform([prediction_encoded])[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Traffic Situation: {prediction}")

    total = car_count + bike_count + bus_count + truck_count

    st.write("Total vehicle count:", total)

    if prediction == "heavy":
        st.warning("Traffic is heavy. Congestion control may be needed.")
    elif prediction == "high":
        st.warning("Traffic is high. Please monitor the road condition.")
    elif prediction == "normal":
        st.info("Traffic is normal.")
    else:
        st.success("Traffic is low.")


# Optional dataset preview
st.header("Dataset Preview")

if st.checkbox("Show dataset"):
    df = pd.read_csv("Traffic.csv")
    st.dataframe(df.head(20))