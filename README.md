# AI-Powered Traffic Analytics Dashboard

An interactive Streamlit dashboard that combines machine learning, traffic prediction, and OpenAI-powered mobility analytics.

This project was developed to explore how AI-assisted workflows can support traffic analysis, congestion interpretation, and mobility-related decision making.

---

## Project Overview

The application allows users to:

- Input traffic conditions manually
- Predict traffic situations using a Decision Tree machine learning model
- Interact with an AI-powered Traffic Analyst Assistant
- Receive natural-language explanations and mobility-related recommendations
- Explore conversational traffic insights with chat history support

The project combines traditional machine learning with large language model (LLM) integration to create an AI-assisted traffic analytics workflow.

---

## Features

- Traffic situation prediction using a trained Decision Tree classifier
- Interactive Streamlit dashboard interface
- Vehicle traffic input for:
  - Cars
  - Bikes
  - Buses
  - Trucks
- AI-powered Traffic Analyst Assistant using OpenAI API
- Conversational AI interaction with chat history
- Mobility-related congestion analysis and recommendations
- Natural-language interpretation of prediction results

---

## Dashboard Preview

### Main Dashboard

![Dashboard Screenshot](Screenshot%202026-05-24%20at%2007.33.12.png)

---

### AI Traffic Analyst Assistant

![AI Assistant Screenshot](Screenshot%202026-05-24%20at%2007.33.03.png)

---

## AI Traffic Analyst Assistant

The dashboard includes an AI-powered assistant that allows users to ask natural-language questions about traffic situations and mobility patterns.

The assistant can:

- Explain predicted traffic conditions
- Analyze vehicle composition
- Identify possible congestion trends
- Suggest mobility-related improvements
- Support traffic interpretation with conversational context

### Example Questions

- "What does this traffic situation mean?"
- "Which vehicle type contributes most to traffic volume?"
- "What could mobility planners learn from this result?"
- "Give one recommendation to reduce congestion."

---

## Workflow

The system combines machine learning prediction with AI-powered analysis.

```text
User Input
    ↓
Traffic Prediction Model
    ↓
Prediction Result
    ↓
Context Generation
    ↓
OpenAI API
    ↓
AI Traffic Analysis Response
```

---

## Machine Learning Model

Two Decision Tree classification models were tested during development.

- Model 1 included the `Total` vehicle count feature and achieved perfect accuracy.
- Model 2 excluded the redundant `Total` feature to improve interpretability and reduce feature dependency.

The final application uses the more interpretable model configuration for prediction and AI-assisted analysis.

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- OpenAI API
- Matplotlib
- Seaborn
- Pickle
- Git / GitHub

---

## Project Structure

```text
traffic_flow_analysis/
│
├── app.py
├── traffic_model.pkl
├── label_encoder_day.pkl
├── label_encoder_target.pkl
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── Traffic Prediction.ipynb
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jingxu023/traffic_flow_analysis.git
cd traffic_flow_analysis
```

Create and activate a virtual environment:

```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## OpenAI API Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Future Improvements

- Real-time traffic data integration
- CSV upload support
- LangChain-based workflow orchestration
- Automated anomaly detection
- Multi-agent traffic analysis workflows
- Cloud deployment support

