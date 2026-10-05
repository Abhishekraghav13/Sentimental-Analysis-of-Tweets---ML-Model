# Sentimental-Analysis-of-Tweets---ML-Model
# Twitter Sentiment Analysis using Machine Learning

A Machine Learning project that classifies tweets as **Positive** or **Negative** using Natural Language Processing (NLP) and Logistic Regression.

The project uses the **Sentiment140 dataset containing 1.6 million tweets** and applies text preprocessing, stemming, TF-IDF feature extraction, and Logistic Regression for sentiment classification.

---

##  Project Overview

Social media platforms contain massive amounts of textual information that can be analyzed to understand public sentiment.

This project develops a binary sentiment classification model capable of determining whether a given tweet expresses a **positive** or **negative** sentiment.

### Objective

Build an NLP-based Machine Learning pipeline that:

- Processes raw Twitter text
- Removes unnecessary characters and stopwords
- Applies stemming
- Converts text into numerical TF-IDF features
- Trains a Logistic Regression classifier
- Evaluates the model on unseen tweets
- Saves the trained model for future predictions

---

##  Dataset

The project uses the **Sentiment140 dataset**, which contains **1,600,000 tweets** collected through the Twitter API.

The dataset contains six fields:

| Column | Description |
|---|---|
| `target` | Sentiment label |
| `ids` | Tweet ID |
| `date` | Tweet timestamp |
| `flag` | Query information |
| `user` | Twitter username |
| `text` | Tweet content |

### Sentiment Labels

The original dataset uses:

- `0` → Negative
- `4` → Positive

For binary classification, the project converts:

```text
0 → Negative
4 → Positive
```

The dataset is perfectly balanced:

- **800,000 Negative tweets**
- **800,000 Positive tweets**

### Dataset Source

Sentiment140 dataset by Go, Bhayani and Huang.

> Go, A., Bhayani, R. and Huang, L. (2009). *Twitter sentiment classification using distant supervision.*

---

##  Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **NLTK**
- **Scikit-learn**
- **Matplotlib**
- **Jupyter Notebook / Google Colab**
- **Kaggle Dataset**

---

##  Machine Learning Pipeline

```text
Sentiment140 Dataset
        ↓
Data Loading & Exploration
        ↓
Text Preprocessing
        ↓
Stopword Removal
        ↓
Porter Stemming
        ↓
TF-IDF Feature Extraction
        ↓
Train-Test Split
        ↓
Logistic Regression
        ↓
Model Evaluation
        ↓
Saved Trained Model
```

---

##  Data Preprocessing

The raw tweet text undergoes several preprocessing steps.

### 1. Removing Non-Alphabetic Characters

Characters such as:

- `@`
- Numbers
- URLs
- Special characters

are removed using regular expressions.

### 2. Lowercasing

All text is converted to lowercase to maintain consistency.

### 3. Stopword Removal

Common English stopwords are removed using the NLTK stopwords corpus.

### 4. Stemming

The **Porter Stemmer** is used to reduce words to their root form.

For example:

```text
actor
actress
```

are reduced to their stem.

The processed text is stored in a new `stemmed_content` column.

---

##  Feature Extraction

Machine Learning models cannot directly process raw text, so the processed tweets are converted into numerical feature vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

TF-IDF assigns importance to words based on:

- How frequently they occur in a tweet
- How frequently they occur across the complete dataset

This produces a sparse numerical representation of the tweets suitable for Machine Learning.

---

##  Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** as the classification algorithm.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

The model learns patterns in the TF-IDF representations and predicts whether an unseen tweet is positive or negative.

---

##  Train-Test Split

The dataset is divided into:

| Dataset | Samples |
|---|---:|
| Training | 1,280,000 |
| Testing | 320,000 |
| Total | 1,600,000 |

This corresponds to an **80/20 train-test split**.

---

##  Model Performance

The model achieved the following results:

| Dataset | Accuracy |
|---|---:|
| Training | **79.87%** |
| Testing | **77.67%** |

### Final Test Accuracy

**77.67%**

The model demonstrates reasonable generalization to unseen tweets while maintaining a relatively small gap between training and testing accuracy.

---

##  Saving the Model

The trained Logistic Regression model is saved using Python's `pickle` module:

```python
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))
```

The saved model can subsequently be loaded for making predictions without retraining the classifier.

---

##  Prediction

The saved model can be loaded using:

```python
loaded_model = pickle.load(
    open('trained_model.sav', 'rb')
)
```

A processed tweet can then be passed to the model for classification.

The output is interpreted as:

```text
0 → Negative Tweet
1 → Positive Tweet
```

---

##  Project Structure

```text
Twitter-Sentiment-Analysis/
│
├── Twitter_Sentiment_Analysis_using_ML.ipynb
├── trained_model.sav
└── README.md
```

> The dataset is not included in this repository because of its large size. It can be downloaded separately from Kaggle.

---

##  How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Twitter-Sentiment-Analysis
```

### 2. Install dependencies

```bash
pip install numpy pandas nltk scikit-learn matplotlib
```

### 3. Download the Sentiment140 dataset

Download the dataset from Kaggle and place the CSV file in the appropriate project directory.

### 4. Open the notebook

```bash
jupyter notebook Twitter_Sentiment_Analysis_using_ML.ipynb
```

Alternatively, open the notebook using **Google Colab**.

### 5. Run the notebook

Execute the cells sequentially to:

- Load the dataset
- Preprocess tweets
- Generate TF-IDF features
- Train the Logistic Regression model
- Evaluate performance
- Save the trained model

---

##  Key Takeaways

- Worked with a large-scale dataset containing **1.6 million tweets**
- Performed practical NLP preprocessing on social-media text
- Implemented **Porter Stemming** and **stopword removal**
- Used **TF-IDF** for text feature representation
- Built a **Logistic Regression** sentiment classifier
- Achieved **77.67% test accuracy**
- Saved the trained model using `pickle`

---

##  Future Improvements

Potential improvements to the project include:

- Experimenting with **Naive Bayes, SVM, Random Forest and ensemble models**
- Using **n-gram TF-IDF features**
- Hyperparameter tuning
- Evaluating precision, recall and F1-score
- Handling emojis and Twitter-specific language
- Using word embeddings such as Word2Vec or GloVe
- Exploring modern Transformer-based models such as BERT
- Building a real-time sentiment analysis application

---

##  References

**Sentiment140 Dataset**

Go, A., Bhayani, R., & Huang, L. (2009).  
*Twitter Sentiment Classification using Distant Supervision.*

Dataset: `kazanova/sentiment140`

---

##  Author

**Abhishek Raghav**

Machine Learning | Data Analytics | Python | NLP
