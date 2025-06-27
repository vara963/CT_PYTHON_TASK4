# CT_PYTHON_TASK4
A mini NLP project using the Naive Bayes algorithm to detect whether an SMS message is spam or not. Built with Python and scikit-learn, trained on a real-world dataset.

# 📬 SMS Spam Classifier (Naive Bayes - NLP Project)

This project uses a basic Natural Language Processing (NLP) approach to classify SMS text messages as **spam** or **ham** (not spam). It applies the **Naive Bayes algorithm** and simple text vectorization to train the model.

---

## 🔧 Tools & Libraries Used

- Python 🐍
- pandas
- scikit-learn (sklearn)
- CountVectorizer
- MultinomialNB

---

## 📦 Dataset

- Source: [SMS Spam Dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)
- Contains two columns:
  - `label`: spam or ham
  - `message`: the SMS content

---

## 🚀 Steps Performed

1. **Load & Preprocess Data**
2. **Convert Labels** to numerical values (0 = ham, 1 = spam)
3. **Split Data** into training and test sets
4. **Convert Messages to Vectors** using `CountVectorizer`
5. **Train Model** using `MultinomialNB`
6. **Predict & Evaluate** performance

---

## 📈 Accuracy & Evaluation

After training and testing, the model shows:

- ✅ **Accuracy:** ~98%
- 📊 Detailed **classification report** with precision, recall, and f1-score.

---

## 💻 Sample Code Snippet

```python
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vec, y_train)
