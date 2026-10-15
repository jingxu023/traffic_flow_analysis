# Traffic Prediction Dashboard

A Streamlit dashboard for exploring traffic data with a Decision Tree classifier, a time-based traffic-volume comparison, and an optional OpenAI assistant.

I started this project as a machine-learning exercise using vehicle counts from a Kaggle dataset. After testing the original classifier, I added a second mode to answer a different question: **is the entered traffic volume unusually high or low for that day and time?**

The dashboard lets you enter the hour, weekday, and counts of cars, bikes, buses, and trucks. It then analyzes those inputs using either V1 or V2.

## Dashboard

**Traffic input and classification**

![Traffic dashboard](Screenshot%202026-05-24%20at%2007.33.03.png)

**Optional AI Traffic Analyst**

![AI assistant](Screenshot%202026-05-24%20at%2007.33.12.png)

These screenshots show the earlier dashboard interface. The current version also includes the V1/V2 mode selector and V2 reference values.

## Two ways to analyze traffic

### V1 — Decision Tree classification

V1 is a supervised classification model trained on **2,976 records** from `Traffic.csv`. The dataset already provides four `Traffic Situation` labels: `low`, `normal`, `high`, and `heavy`.

The model uses the hour, weekday, and four vehicle counts to classify an input into one of those original labels. It does not generate the labels itself, and it is not forecasting vehicle numbers in the future.

During development, I also tested a model that included the `Total` vehicle count. I then excluded that redundant total from the final model, leaving the individual vehicle counts as inputs.

**Feature importance of the saved V1 Decision Tree (`traffic_model.pkl`):**

| Feature | Importance |
|:--|--:|
| CarCount | **50.30%** |
| BusCount | 19.58% |
| TruckCount | 18.20% |
| Hour | 7.09% |
| BikeCount | 4.18% |
| Day of the week | 0.66% |

The vehicle counts have the largest influence on the tree's splits. Hour and weekday are used too, but much less. Feature importance here measures the relative reduction in node impurity; **it does not describe the causes of congestion**.

### Why I added V2

While testing V1, I noticed that changing the hour or weekday often left the predicted label unchanged when the vehicle counts stayed the same. The feature importance results helped explain this: the Decision Tree relies mainly on vehicle counts when reproducing the original dataset labels.

That isn't necessarily a bug. But I wanted to explore something V1 wasn't designed to answer: **how does a given traffic count compare with what is typical for that weekday and hour?**

### V2 — Relative traffic-volume classification

V2 uses a simple, rule-based comparison instead of training another Decision Tree.

1. Find the historical **median** vehicle count for the selected weekday and hour, using a training-only portion of the dataset.
2. Divide the entered total by that reference value.
3. Assign a relative-volume category: `Low`, `Normal`, `High`, or `Very High`. The boundaries come from quartiles of the training-data ratios.

```text
Relative ratio = entered total vehicles / historical median vehicles
```

For example, 150 vehicles might be lower than usual on a busy morning but well above the normal level late at night. V2 can show that distinction because its reference changes with the selected time.

If a weekday/hour group contains fewer than **8 observations**, V2 falls back to the median for that hour across all weekdays.

These are **relative traffic-volume categories**, not measured levels of congestion. V2 does not use the original Kaggle `Traffic Situation` labels.

### V1 vs. V2

| | V1: Decision Tree | V2: Relative volume |
|:--|:--|:--|
| Question | Which original traffic label fits this input? | Is this count high or low for this weekday and hour? |
| Method | Supervised machine learning | Historical median, ratio, and quartile-based rules |
| Output | `low`, `normal`, `high`, `heavy` | `Low`, `Normal`, `High`, `Very High` |
| Time information | Model features, with relatively low importance | Directly determines the reference value |
| Original Kaggle labels | Used for training | Not used |

## AI Traffic Analyst (optional)

The dashboard includes an assistant powered by the OpenAI API. It receives the current analysis context and can answer follow-up questions in natural language, for example:

- “What does this traffic situation mean?”
- “Which vehicle type contributes most to the total?”
- “Why does V2 rate this time differently?”

The assistant is for interpreting results and discussing possible mobility implications. Its answers are **not independent measurements or verified traffic forecasts**. The dashboard's V1 and V2 features work without an API key.

## Run with Docker

From the project directory, with Docker running:

```bash
git clone https://github.com/jingxu023/traffic_flow_analysis.git
cd traffic_flow_analysis
docker compose up --build -d
```

Open **http://localhost:8501**.

```bash
# See application logs
docker compose logs -f

# Stop the service
docker compose down

# Rebuild after code changes
docker compose up --build -d
```

The Docker image includes the saved V1 model, encoders, and V2 reference data. The Streamlit health endpoint is at `http://localhost:8501/_stcore/health`.

### Enable the OpenAI assistant

Create a private `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and replace the example value with your own valid API key:

```env
OPENAI_API_KEY=your_actual_api_key
```

Then recreate the container so it receives the updated environment variable:

```bash
docker compose up -d --force-recreate
```

**Never commit `.env` or share your key.** An invalid or missing key only affects the optional assistant, not V1 or V2.

## Run without Docker

```bash
git clone https://github.com/jingxu023/traffic_flow_analysis.git
cd traffic_flow_analysis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

On Windows, use `.venv\\Scripts\\activate` instead of the `source` command.

## Tests and development workflow

The project includes input validation, logging, automated tests, linting with Ruff, and a GitHub Actions workflow that checks the code and builds a Docker image.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check app.py train_model.py src tests
```

To regenerate the V2 reference values from the CSV:

```bash
python build_weekday_v2.py --csv Traffic.csv
```

## Main files

| File | Role |
|:--|:--|
| `app.py` | Streamlit interface, mode selector, and AI assistant |
| `train_model.py` | V1 Decision Tree training |
| `src/prediction.py` | V1 feature processing and predictions |
| `src/relative_weekday.py` | V2 weekday/hour reference lookup and classification |
| `src/relative_traffic.py` | Reusable hour-based relative-volume logic |
| `build_weekday_v2.py` | Generates the V2 weekday/hour reference file |
| `relative_traffic_weekday_v2.json` | Saved reference medians and category cutoffs |
| `tests/` | Prediction and relative-volume tests |
| `Dockerfile`, `docker-compose.yml` | Container setup |
| `.github/workflows/ci.yml` | CI checks |

## Limitations

This project uses a small historical dataset, not a live traffic feed.

- The original dataset's four V1 labels are provided, but their labeling criteria have not been independently verified.
- V2 uses the dataset's weekday labels. Because full month/year dates are missing and parts of the date sequence are inconsistent, those weekday labels cannot be independently checked against a calendar.
- Each weekday/hour reference is based on a limited number of recorded days. Observations from the same day are not fully independent.
- Neither version has traffic-speed or road-capacity measurements, so neither can confirm a traffic jam. Neither version predicts future traffic volume.

V2 grew out of a practical question I ran into while testing V1. Keeping both modes makes it easier to see the difference between learning existing labels and making a transparent comparison against a historical baseline.
