# Sentiment Analysis of Yelp Reviews at Scale

Big Data project using **Apache Spark (PySpark)** and **NLP** to classify millions of Yelp reviews as **Positive** or **Negative**.

- **Dataset:** [Yelp Open Dataset (Kaggle)](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) — ~7M reviews, ~5 GB of text
- **Labels:** 4–5 stars → Positive (1), 1–2 stars → Negative (0), 3 stars removed
- **Environment:** Google Colab + Google Drive

## Project Phases

| Phase | Notebook | Status |
|---|---|---|
| 1. Data Acquisition & Preparation | `01_data_preparation.ipynb` | ✅ Done |
| 2. EDA & Text Preprocessing | `02_eda_preprocessing.ipynb` | ⏳ To do |
| 3. Feature Engineering & Modeling | `03_modeling.ipynb` | ⏳ To do |
| 4. Evaluation, Insights & Reporting | `04_evaluation.ipynb` | ⏳ To do |

## Phase 1 Results

| Metric | Value |
|---|---|
| Total raw reviews | 6,990,280 |
| Removed (3-star neutral: 691,934 + short/duplicate: 167) | 692,101 |
| Reviews after cleaning | 6,298,179 |
| Positive (label 1) | 4,684,444 (74.4%) |
| Negative (label 0) | 1,613,735 (25.6%) |
| Imbalance ratio (positive : negative) | 2.90 : 1 |
| Year range | 2005 – 2022 (2022 is partial, ~29K reviews) |
| Businesses | 150,346 |

**Key observations**
- Data is imbalanced (~3:1) → Phase 3 must use class weights; evaluate with F1 / Recall, not only Accuracy.
- Review volume grows every year until 2019, drops in 2020 (COVID-19), and recovers in 2021.
- 2022 contains only a few weeks of data → handle carefully in time-based analysis.

**Cleaned data schema** (`reviews_clean.parquet`)
- `review_id`
- `business_id`
- `stars`
- `text`
- `date`
- `label`
- `year`

## Getting the Data (for team members)

The cleaned data is **not** stored in this repo (it is several GB). It lives on Google Drive in the shared `yelp_project` folder:

```
yelp_project/
├── reviews_clean.parquet   # review_id, business_id, stars, text, date, label, year
└── business.parquet        # business_id, name, city, state, categories
```

1. Open the shared `yelp_project` folder link.
2. Right-click the folder → **Organize → Add shortcut** → **My Drive**.
   This makes it appear at `/content/drive/MyDrive/yelp_project` in Colab.
3. Start your notebook with:

```python
!pip install -q pyspark
from google.colab import drive
drive.mount('/content/drive')

from pyspark.sql import SparkSession
spark = (SparkSession.builder
         .appName("YelpSentiment")
         .config("spark.driver.memory", "10g")
         .getOrCreate())

df       = spark.read.parquet("/content/drive/MyDrive/yelp_project/reviews_clean.parquet")
business = spark.read.parquet("/content/drive/MyDrive/yelp_project/business.parquet")

# while developing: work on a 5% sample, then run the final version on all data
sample = df.sample(fraction=0.05, seed=42)
```

You do **not** need to re-run Phase 1 unless you want to rebuild the data from scratch.

## Rules
- Never commit `kaggle.json` or any data files.
- Clear large outputs before committing notebooks.
- Name notebooks with the phase number prefix (`02_…`, `03_…`).
