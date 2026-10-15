# Traffic Prediction Dashboard

A Streamlit application that classifies traffic conditions using a Decision Tree model and provides an optional AI assistant for explaining the results. I built it to practice working with a machine-learning model in a usable application, from data preparation and testing to Docker deployment.

## What the app does

Users can select an hour and weekday, enter the number of cars, bikes, buses, and trucks, and get a traffic classification. The main mode (V1) uses a Decision Tree trained on a Kaggle traffic dataset. The app also includes an optional OpenAI-powered assistant for questions about the result and the vehicle counts.

The current dashboard offers two modes:

- **V1 — Decision Tree:** predicts one of the original dataset's four labels: `low`, `normal`, `high`, or `heavy`.
- **V2 — Relative traffic volume:** compares the entered total with typical traffic volume at the selected weekday and hour. This is a small additional analysis feature, not a second machine-learning model.

## Dashboard previews

**Traffic dashboard**

![Traffic dashboard](Screenshot%202026-05-24%20at%2007.33.03.png)

**AI Traffic Analyst**

![AI Traffic Analyst](Screenshot%202026-05-24%20at%2007.33.12.png)

*These screenshots are from an earlier version of the interface.*

## Machine learning: V1

I trained a `DecisionTreeClassifier` on 2,976 records from `Traffic.csv`, using an 80/20 training/test split. The target column, `Traffic Situation`, already contains the four traffic labels in the original dataset. The model learns to classify those labels from six input features: hour, weekday, and the counts of cars, bikes, buses, and trucks.

During development, I also tried using the `Total` vehicle-count column. I excluded it from the final input features because it duplicates information already present in the individual vehicle counts.

### What I learned from feature importance

When I tested the dashboard, changing the hour or weekday often didn't change the predicted class if I kept the vehicle counts the same. I checked the saved model's `feature_importances_` to understand why.

| Feature | Importance |
| --- | ---: |
| CarCount | **50.30%** |
| BusCount | 19.58% |
| TruckCount | 18.20% |
| Hour | 7.09% |
| BikeCount | 4.18% |
| Day of the week | 0.66% |

Vehicle counts clearly dominate the Decision Tree's splits. The model **does** use hour and weekday in some branches, but their overall importance is much lower. This helps explain why changing the time does not always change the output.

Feature importance describes how much each feature contributes to reducing impurity in this particular tree. It does not tell us what *causes* traffic congestion.

This also raised a useful question for me: if the original labels depend mostly on vehicle counts, would a simpler set of rules work almost as well? I have not established that comparison yet, so I would not claim the Decision Tree is necessarily better than a rule-based baseline.

## Small extension: V2

After looking into V1, I added a second mode to answer a slightly different question: **Is this vehicle count high or low compared with the usual volume at this weekday and hour?**

V2 compares the entered total with the historical median for the selected weekday and hour, then assigns `Low`, `Normal`, `High`, or `Very High` using thresholds derived from the training data. If a weekday/hour group has fewer than eight observations, it uses an hour-only reference instead. V2 does not use the original Kaggle traffic labels or another Decision Tree.

It is a relative traffic-volume comparison, **not** a measurement of actual traffic jams or a forecast of future traffic. The dataset also lacks complete calendar dates, so its weekday labels cannot be independently verified.

## AI Traffic Analyst

The optional assistant uses the OpenAI API to interpret the current dashboard result in plain language. Users can ask, for example:

- “What does this traffic situation mean?”
- “Which vehicle type contributes most to the total?”
- “What could mobility planners consider based on this result?”

The assistant receives the current analysis context and supports a short conversation history. Its answers are explanations and suggestions, not independently verified traffic measurements. **Both classification modes work without an API key.**

## Tech stack

- **Python, Pandas, scikit-learn** — data preparation and Decision Tree classification
- **Streamlit** — interactive dashboard
- **OpenAI API** — optional conversational assistant
- **Pytest, Ruff** — tests and linting
- **Docker, Docker Compose** — reproducible runtime
- **GitHub Actions** — automated checks and image build

## Run with Docker

With Docker running:

```bash
git clone https://github.com/jingxu023/traffic_flow_analysis.git
cd traffic_flow_analysis
docker compose up --build -d
```

Open [http://localhost:8501](http://localhost:8501).

```bash
# Follow logs
docker compose logs -f

# Stop containers
docker compose down
```

The Docker image includes the trained model, label encoders, and V2 reference data. The Streamlit health endpoint is available at `http://localhost:8501/_stcore/health`.

### Optional OpenAI API key

To enable the assistant, copy the example environment file:

```bash
cp .env.example .env
```

Set your own valid API key inside `.env`:

```env
OPENAI_API_KEY=your_actual_api_key
```

Then recreate the container:

```bash
docker compose up -d --force-recreate
```

Do not commit `.env` or share your API key.

## Run locally

```bash
git clone https://github.com/jingxu023/traffic_flow_analysis.git
cd traffic_flow_analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

## Tests and DevOps

I separated model loading and prediction from the dashboard UI, added input validation and logging, and wrote tests for the prediction and relative-volume logic. GitHub Actions runs linting, tests, and a Docker image build.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check app.py train_model.py src tests
```

To regenerate V2's reference values from `Traffic.csv`:

```bash
python build_weekday_v2.py --csv Traffic.csv
```

## Main files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI and AI assistant |
| `train_model.py` | Decision Tree training |
| `src/prediction.py` | Model loading, inputs, and predictions |
| `traffic_model.pkl` | Saved V1 classifier |
| `src/relative_weekday.py` | V2 relative traffic-volume logic |
| `relative_traffic_weekday_v2.json` | V2 historical references and cutoffs |
| `tests/` | Automated tests |
| `Dockerfile`, `docker-compose.yml` | Container setup |
| `.github/workflows/ci.yml` | CI workflow |

## Limitations and next steps

The dataset is small and historical. The original four classification labels come from Kaggle, and I have not independently verified how they were assigned. Neither mode uses live traffic speeds or road capacity, so the labels should not be treated as evidence of actual congestion or as a future-traffic forecast.

Possible next steps are comparing the Decision Tree with a simple rule-based baseline, evaluating performance on more varied data, and integrating reliable real-time traffic information.
