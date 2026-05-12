# Alzheimer's Disease Detection using Machine Learning

A machine learning project that predicts whether a patient has Alzheimer's Disease based on clinical health data — built as part of an ML course project (May 2026).

---

## About the Project

This project uses a real-world clinical dataset of 2,149 patients with 35 features — things like age, BMI, memory complaints, cognitive test scores, lifestyle habits, and medical history — to build a model that flags patients likely to have Alzheimer's Disease.

The idea is simple: if we can detect Alzheimer's early from routine health checkup data, doctors can act faster and improve patient outcomes.

- **Domain:** Healthcare / Biological Data  
- **Type:** Binary Classification — Diagnosed (1) vs Not Diagnosed (0)  
- **Dataset:** [Alzheimer's Disease Dataset by Rabie El Kharoua — Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

---

## What's in this repo

```
alzheimers-disease-detection/
│
├── alzheimers_disease_detection.ipynb   # Main project notebook (Google Colab)
└── README.md
```

> **Note:** The dataset (`alzheimers_disease_data.csv`) is not included here because of file size. Download it from the Kaggle link above.

---

## Project Steps Covered

| Step | What |
|------|------|
| Step 1 | Load & understand the data |
| Step 2 | Data cleaning, scaling (StandardScaler), class balancing (SMOTE) |
| Step 3 | Exploratory Data Analysis — plots, trends, insights |
| Step 4 | Baseline model (Logistic Regression) + Feature Selection |
| Step 5 | Model improvement — Random Forest & XGBoost |
| Step 6 | Hyperparameter tuning + Regularization |
| Step 7 | Final comparison table + Streamlit deployment (bonus) |

---

## How to Run This

### Option 1 — Google Colab (Recommended, easiest)

1. Open the notebook directly in Colab:  
   Click the `.ipynb` file in this repo → click **"Open in Colab"** button at the top

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) and upload it to your **Google Drive** in a folder called `alzheimers-disease-detection`

3. In the notebook, update this line to match your Drive folder path:
   ```python
   df = pd.read_csv('/content/drive/MyDrive/alzheimers-disease-detection/alzheimers_disease_data.csv')
   ```

4. Run all cells from top to bottom — `Runtime → Run all`

---

### Option 2 — Run Locally (Jupyter Notebook)

If you want to run it on your own machine instead of Colab:

**Step 1 — Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

**Step 2 — Download the dataset**  
Get `alzheimers_disease_data.csv` from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) and put it in the same folder as the notebook.

**Step 3 — Update the file path in the notebook**  
Change the CSV loading line from the Colab Drive path to a local path:
```python
# Change this:
df = pd.read_csv('/content/drive/MyDrive/alzheimers-disease-detection/alzheimers_disease_data.csv')

# To this:
df = pd.read_csv('alzheimers_disease_data.csv')
```

Also remove or skip the Google Drive mount cell:
```python
# Skip this cell if running locally
from google.colab import drive
drive.mount('/content/drive')
```

**Step 4 — Launch Jupyter and open the notebook**
```bash
jupyter notebook
```

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` & `seaborn` | Visualizations |
| `scikit-learn` | Models, scaling, train-test split, metrics |
| `imbalanced-learn` | SMOTE for class balancing |
| `xgboost` | XGBoost classifier |

---

## Key Results

| Model | F1-Score | ROC-AUC |
|-------|----------|---------|
| Logistic Regression (baseline) | 0.82 | 0.89 |
| After Feature Selection | 0.83 | 0.89 |
| Random Forest | — | — |
| XGBoost | — | — |
| After Tuning | — | — |

*(Results table will be updated as steps are completed)*

---

## Author

**Vishaal** — ML Course Project, May 2026
