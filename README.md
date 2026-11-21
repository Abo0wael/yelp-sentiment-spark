# Sentiment Analysis of Yelp Reviews at Scale

Big Data project using **Apache Spark (PySpark)** and **NLP** to classify millions of Yelp reviews as **Positive** or **Negative**.

- **Dataset:** [Yelp Open Dataset (Kaggle)](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) — ~7M reviews, ~5 GB of text
- **Labels:** 4–5 stars → Positive (1), 1–2 stars → Negative (0), 3 stars removed
- **Environment:** Google Colab + Google Drive

## Project Phases

| Phase | Deliverable | Status |
|---|---|---|
| 1. Proposal & Dataset | [`PROPOSAL.md`](PROPOSAL.md), `01_data_preparation.ipynb` | ✅ Done |
| 2. Data Cleaning & Spark Processing | `02_spark_processing.ipynb` | ✅ Done |
| 3. EDA & Machine Learning | EDA started in `02_eda_preprocessing.ipynb`; ML notebook to come | ⏳ To do |
| 4. Interactive Dashboard (Streamlit) | Streamlit app (reads `dashboard_data/`) | ⏳ To do |
| 5. Report & Presentation | Final report + slides | ⏳ To do |

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

## Phase 2 Results

All numbers come from `02_spark_processing.ipynb` (Spark 4.0.5 on Colab). Aggregations, joins and SQL ran on the **full** data (6,298,179 reviews); text statistics on the **5% sample** (314,338 reviews, `seed=42`).

### Data quality & missing values

| Check | Reviews | Business |
|---|---|---|
| Null values | 0 (all 7 columns) | 103 (all in `categories`) → filled with `"Unknown"` |
| Empty strings | 0 empty text | 0 categories, 0 city, 0 name |
| Duplicate IDs | 0 `review_id` | 0 `business_id` |
| Unexpected stars / label mismatch | 0 / 0 | – |
| Reviews without a matching business (`left_anti` join) | 0 | – |
| Date range | 2005-02-16 → 2022-01-19 | – |

### Key aggregation findings

**Yearly trend**: volume peaks in 2019 (832,821 reviews), drops in 2020 (517,544) and partly recovers in 2021 (574,499). The positive share falls steadily from 87.7% (2005) to 70.6% (2021).

| Period | Reviews per year | Positive share |
|---|---|---|
| 2015–2019 (pre-COVID) | 741,948 | 74.27% |
| 2020 (COVID-19) | 517,544 | 72.87% |
| 2021 (recovery) | 574,499 | 70.59% |

**Review length**: negative reviews are longer (avg / median / p95 words: **134.1 / 98 / 368**) than positive ones (**91.8 / 66 / 251**). Reviews over 500 words (56,681) are only 41.6% positive, and reviews mentioning "refund" (49,284) only 8.8%.

**Top cities by reviews**

| City | Reviews | Positive share |
|---|---|---|
| Philadelphia, PA | 853,792 | 77.09% |
| New Orleans, LA | 567,808 | 81.09% |
| Tampa, FL | 411,866 | 74.23% |
| Nashville, TN | 405,129 | 76.61% |
| Tucson, AZ | 368,433 | 72.20% |

**Most negative categories** (≥ 50,000 reviews; 102 of 1,312 categories qualify)

| Category | Reviews | Positive share |
|---|---|---|
| Car Dealers | 63,641 | 47.73% |
| Real Estate | 53,540 | 49.04% |
| Fast Food | 207,340 | 52.80% |
| Auto Parts & Supplies | 59,265 | 53.61% |
| Hotels | 165,313 | 58.68% |

Most positive: Tours (88.49%), Vegan (86.93%), Food Trucks (86.87%).

### Text preprocessing pipeline (5% sample)

`SQLTransformer` (clean) → `RegexTokenizer` → `StopWordsRemover` (keeps negations / intensifiers) → `NGram(2)`; saved to `models/preprocessing_pipeline`.

| Metric | Value |
|---|---|
| Avg tokens per review: before → after stop words | 97.3 → 54.6 (43.9% of tokens removed) |
| Vocabulary size: before → after stop words | 118,778 → 118,665 |
| Handmade checks (`not` kept, URLs removed, `not good` bigram) | passed |

### Spark features used
- **DataFrames & Parquet I/O**: schemas, reading/writing Parquet on Drive
- **Missing-value handling**: one-pass null counts, empty-string checks, `when()` fill with `"Unknown"`
- **Transformations**: `withColumn`, `select` / `drop`, `filter`, `between`, `when`
- **Aggregations**: `groupBy` / `agg`, `countDistinct`, `percentile_approx`
- **Joins**: `left_anti` (key check), `inner` with `broadcast`, `left_semi`
- **Arrays**: `split` + `explode` for categories
- **Spark SQL**: temp views, `JOIN`, `HAVING`, `LATERAL VIEW`, CTEs; **4 / 4 queries match the DataFrame API exactly**
- **Window functions**: `RANK() OVER (PARTITION BY state …)`
- **Spark ML Pipeline**: `SQLTransformer`, `RegexTokenizer`, `StopWordsRemover`, `NGram`
- **Caching / sampling**: `cache()` / `unpersist()`, `sample(0.05, seed=42)`
- **Dashboard export**: `yearly_stats` (18 rows), `city_stats` (50), `category_stats` (1,312), `length_stats` (2) in `dashboard_data/`

## Phase 3 EDA: early results (`02_eda_preprocessing.ipynb`) — In Progress

### Task 1: Star & Sentiment Distribution
- **Star Counts:**
  - 1-star: 1,069,507 reviews
  - 2-star: 544,228 reviews
  - 4-star: 1,452,876 reviews
  - 5-star: 3,231,568 reviews
- **Sentiment Breakdown:** Positive (74.4%) vs. Negative (25.6%), reflecting an imbalance ratio of ~2.90 : 1.
- **Figure:** `figures/01_label_distribution.png`

### Task 2: Review Length Analysis
Word counts calculated on PySpark across sentiment classes:

| Metric | Negative Reviews (Label 0) | Positive Reviews (Label 1) |
|---|---|---|
| Average Word Count | **134.1 words** | **91.8 words** |
| Median Word Count | **98.0 words** | **66.0 words** |
| 95th Percentile | **368.0 words** | **251.0 words** |

- **Observation:** Dissatisfied customers write considerably longer and more detailed reviews when detailing complaints, whereas satisfied customers tend to write shorter feedback.
- **Figure:** `figures/02_review_length.png`

### Task 3: Reviews and Sentiment Over Time (2005 – 2022)
- Steady growth from 2005 (683 reviews) peaking in 2019 (832,821 reviews).
- Significant dip in 2020 (517,544 reviews) due to the COVID-19 pandemic, followed by a rebound in 2021 (574,499 reviews).
- Positive sentiment share declined gradually over the years from ~87–89% in early platform years (2005–2006) to ~70.6% in 2021.
- **Figure:** `figures/03_reviews_per_year.png`

## Getting the Data (for team members)

The cleaned data is **not** stored in this repo (it is several GB). It lives on Google Drive in the shared `yelp_project` folder:

```
yelp_project/
├── reviews_clean.parquet          # review_id, business_id, stars, text, date, label, year   (Phase 1)
├── business.parquet               # business_id, name, city, state, categories             (Phase 1)
├── models/
│   └── preprocessing_pipeline/    # fitted Spark ML text pipeline                           (Phase 2)
└── dashboard_data/                # small summary tables for Streamlit                      (Phase 2)
    ├── yearly_stats/
    ├── city_stats/
    ├── category_stats/
    └── length_stats/
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

## How to run

1. Open the notebook in **Google Colab** (File → Open notebook → GitHub, or upload it).
2. Choose **Runtime → Change runtime type → CPU** with **High-RAM** (no GPU needed; Spark uses the CPU).
3. Make sure the shared `yelp_project` folder is in your Drive (see *Getting the Data*).
4. Run the notebooks **in order**, each one top to bottom (**Runtime → Run all**):

| Order | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | `01_data_preparation.ipynb` | Kaggle (needs your Kaggle API token) | `reviews_clean.parquet`, `business.parquet` |
| 2 | `02_spark_processing.ipynb` | the two Parquet files | `models/preprocessing_pipeline/`, `dashboard_data/` |

Step 1 only has to be run once for the whole team. If the Parquet files already exist on Drive, start directly with step 2.

## Rules
- Never commit `kaggle.json` or any data files.
- Clear large outputs before committing notebooks.
- Name notebooks with the phase number prefix (`02_…`, `03_…`).
