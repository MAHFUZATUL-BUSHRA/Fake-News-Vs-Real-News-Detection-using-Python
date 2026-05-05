# 📰 Fake-News-Vs-Real-News-Detection-using-Python

A machine learning project that detects fake news using Natural Language Processing (NLP) and multiple classification algorithms trained on labeled news article data.

---

## 📌 Project Overview

This project builds and compares several ML models to classify news articles as **FAKE** or **REAL**. It covers the full pipeline — from raw text preprocessing to model tuning and evaluation.

---

## 🗂️ Dataset

- **Format:** CSV file with `text` and `label` columns
- **Labels:** `FAKE` → 0, `REAL` → 1
- **Loaded via:** Google Colab file upload
- link: https://drive.google.com/file/d/1gdZqOKWSMFeXyY6Qka0iyHvf53I51qAT/view?usp=sharing
---

## 🔧 Tech Stack

| Category       | Libraries                                      |
|----------------|------------------------------------------------|
| Data           | `pandas`, `numpy`                              |
| Visualization  | `matplotlib`, `seaborn`                        |
| NLP            | `nltk`, `scikit-learn` (TF-IDF, BoW)          |
| Modeling       | `scikit-learn`, `xgboost`                      |
| Evaluation     | `accuracy_score`, `roc_auc_score`, `confusion_matrix` |

---

## 🚀 Pipeline

### 1. Exploratory Data Analysis (EDA)
- Label distribution plot (Fake vs Real)
  ![p](https://github.com/MAHFUZATUL-BUSHRA/Fake-News-Vs-Real-News-Detection-using-Python/blob/main/images/feke%20vs%20real%20news.png)
- Text length distribution histogram
  ![p](https://github.com/MAHFUZATUL-BUSHRA/Fake-News-Vs-Real-News-Detection-using-Python/blob/main/images/text%20length%20distribution.png)
- Top 20 most frequent words in fake vs real news

### 2. Text Preprocessing
- Lowercasing, URL removal, HTML tag removal
- Punctuation and digit stripping
- Stopword removal using NLTK

### 3. Feature Engineering
- **Bag of Words (BoW)** with `CountVectorizer`
- **TF-IDF** with unigrams + bigrams (`ngram_range=(1,2)`)
- Extra features: `text_length`, `word_count`, `stopword_count`
- Feature combination using `scipy.sparse.hstack`
- Feature scaling with `StandardScaler`

### 4. Models Trained
- Logistic Regression
- Naive Bayes (Multinomial)
- Support Vector Machine (LinearSVC)
- Random Forest
- XGBoost

### 5. Hyperparameter Tuning
All models were tuned using `GridSearchCV` with 3–5 fold cross-validation.

### 6. Evaluation Metrics
- Accuracy score
- Classification report (Precision, Recall, F1)
- Confusion matrix
- ROC-AUC scores
- ROC curve comparison plot

---

## 📊 Results

### Before Fine-Tuning (Baseline Models)
![P](https://github.com/MAHFUZATUL-BUSHRA/Fake-News-Vs-Real-News-Detection-using-Python/blob/main/images/Before%20Scalling.png)

### After Fine-Tuning (GridSearchCV Tuned)
![p](https://github.com/MAHFUZATUL-BUSHRA/Fake-News-Vs-Real-News-Detection-using-Python/blob/main/images/after%20fine%20tuning.png)

### Tuned Models Comparison
![p](https://github.com/MAHFUZATUL-BUSHRA/Fake-News-Vs-Real-News-Detection-using-Python/blob/main/images/model%20comparison.png)

### ROC Curve Comparison
![p](https://github.com/MAHFUZATUL-BUSHRA/Fake-News-Vs-Real-News-Detection-using-Python/blob/main/images/ROC%20curve%20comparison.png)

---

## 🔍 Key Observations

- SVM achieved the highest baseline accuracy (94.08%) before hyperparameter tuning, making it the strongest out-of-the-box model.
- After fine-tuning, XGBoost and SVM tied at 93.69%, showing that tuning helped XGBoost close the gap significantly.
- Logistic Regression and SVM both scored AUC = 0.99, indicating excellent ability to distinguish fake from real news across all classification thresholds.
- Naive Bayes saw the largest accuracy drop after tuning (93% → 84%), suggesting it is more sensitive to hyperparameter changes and may not generalize as well on this dataset.
- All models achieved AUC ≥ 0.97, confirming that the TF-IDF feature representation captures strong discriminative signals across every classifier.
- The ROC curves cluster tightly near the top-left corner, indicating that all models perform well above random chance with very low false positive rates.

---

## 🛠️ How to Run

1. Open the notebook in **Google Colab**
2. Run all cells in order
3. When prompted, upload your CSV dataset file
4. The notebook will preprocess, train, tune, and evaluate all models automatically

---

## 📁 File Structure

```
fake_vs_real_news.ipynb   # Main Jupyter Notebook
README.md                 # Project documentation
```

---

## 📦 Requirements

Install dependencies if running locally:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk xgboost scipy
```

Then download NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

---

## 🙋 Author

> Mahfuzatul Bushra, Data Analyst

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
