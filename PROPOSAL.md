# Project Proposal: Sentiment Analysis of Yelp Reviews at Scale

**Course:** `[Course name / code]` · **Instructor:** `[Dr. name]` · **Semester:** `[Term, Year]`

## Team Members

| # | Name | Student ID |
|---|---|---|
| 1 | `[Name]` | `[ID]` |
| 2 | `[Name]` | `[ID]` |
| 3 | `[Name]` | `[ID]` |
| 4 | `[Name]` | `[ID]` |
| 5 | `[Name]` | `[ID]` |
| 6 | `[Name]` *(optional)* | `[ID]` |
| 7 | `[Name]` *(optional)* | `[ID]` |

## Problem Statement
Online reviews strongly affect which businesses customers choose, but platforms receive millions of them, far too many to read manually. We will build a system that **automatically classifies Yelp reviews as Positive or Negative** from their text, and use it to show how customer sentiment differs across years, cities and business categories. This helps businesses detect dissatisfaction early and helps analysts follow opinion trends at scale.

## Dataset
| | |
|---|---|
| **Name** | Yelp Open Dataset |
| **Link** | https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset |
| **Source / attribution** | Yelp Inc., distributed via Kaggle |
| **License** | Yelp Dataset User Agreement, academic / non-commercial use only |
| **Size** | ~4 GB compressed, **~8.7 GB unzipped**; **6,990,280 reviews**, **150,346 businesses** (2005–2022) |
| **Files used** | `yelp_academic_dataset_review.json` (5.0 GB: review text, stars, date) and `yelp_academic_dataset_business.json` (114 MB: name, city, state, categories) |

**Labels:** 4–5 stars → Positive (1), 1–2 stars → Negative (0); 3-star (neutral) reviews are removed.

## Why Big Data, and Why Spark
- **Volume:** about 7 million reviews and 5 GB of raw text. This is too large to process comfortably with pandas in a single Python process, and text processing multiplies the data (millions of reviews × hundreds of tokens).
- **Variety:** unstructured text plus structured fields (stars, dates, locations, categories) that must be joined.
- **Apache Spark (PySpark)** processes the data in parallel across all CPU cores, keeps intermediate results in memory, and offers one engine for every step: DataFrames, Spark SQL, joins and aggregations, and **Spark ML** for text pipelines and classification. The same code would scale to a real cluster without changes.

## Planned Methodology
1. **Proposal & dataset:** select the dataset; download it with the Kaggle API; clean it, label it and store it as Parquet on Google Drive.
2. **Data cleaning & Spark processing:** inspect data quality, handle missing values; use DataFrame transformations, aggregations, joins, Spark SQL and window functions; build the text preprocessing pipeline.
3. **EDA & machine learning:** explore distributions and trends; extract TF-IDF features; train and compare classifiers (e.g. Logistic Regression, Naive Bayes, Linear SVM) with class weights for the ~3:1 imbalance.
4. **Interactive dashboard:** a Streamlit app showing sentiment by year, city and category, plus live prediction of new reviews.
5. **Report & presentation:** document methods, results and insights.

## Expected Outputs
- A trained **sentiment classification model** (Spark ML pipeline) saved for reuse.
- **Evaluation metrics:** Accuracy, Precision, Recall, **F1-score** and ROC-AUC, with a confusion matrix (F1 / Recall matter because the classes are imbalanced).
- A **Streamlit dashboard** with sentiment insights and a review-prediction demo.
- Reproducible **Colab notebooks** on GitHub, a final report and a presentation.

**Environment:** Google Colab (CPU + High-RAM), Google Drive for data, GitHub for code.
