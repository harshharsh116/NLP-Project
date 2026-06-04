# Module 11: Natural Language Processing
## Project 1: NewsBot – Headline Classifier for InfoStream News

### Load the news headlines and categories from a JSON file.
```python
import pandas as pd
import json

data = []
with open('News_Category_Dataset_v3.json', 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
```

### 1. Keep only these categories: TECHNO, ENTERTAINMENT, POLITICS, BUSINESS.
```python
categories_to_keep = ['TECHNO', 'ENTERTAINMENT', 'POLITICS', 'BUSINESS']
df = df[df['category'].isin(categories_to_keep)]
df = df.reset_index(drop=True)
```

### 2. Load spaCy’s English model without unnecessary components.
```python
import spacy

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
```

### 3. Write a function to preprocess headlines: lowercase, remove stopwords, punctuation, and lemmatize
```python
def preprocess_text(text):
    doc = nlp(text.lower())
    clean_words = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            clean_words.append(token.lemma_)
    return ' '.join(clean_words)
```

### 4. Apply preprocessing to all headlines.
```python
df['clean_headline'] = df['headline'].apply(preprocess_text)
```

### 5. Convert text into numeric vectors using CountVectorizer with unigrams and bigrams
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
X_vectors = vectorizer.fit_transform(df['clean_headline'])
```

### 6. Limit the vocabulary size to 5000 features; explain why
**Explanation:** Limiting the vocabulary size to 5000 features reduces memory consumption, speeds up model training, and prevents overfitting by ignoring rare or less significant words and focusing only on the most common and informative ones.

### 7. Create feature matrix X and label vector y.
```python
X = X_vectors
y = df['category']
```

### 8. Split data into training and test sets with balanced categories.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 9. Train Logistic Regression model with enough iterations.
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
```

### 10. Test the model and report accuracy.
```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))
```

### 11. Build a function to predict categories for new headlines.
```python
def predict_category(headline):
    clean = preprocess_text(headline)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]
    return prediction
```

### 12. Create a Chabot that takes user input and predicts the category until user types 'quit' or 'exit'.
```python
def newsbot_chatbot():
    while True:
        headline = input("Enter headline: ").strip()
        if headline.lower() in ['quit', 'exit']:
            break
        if not headline:
            continue
        category = predict_category(headline)
        print(f"Predicted Category: {category}")
```

### 13. Also, Deploy Streamlit App.
```python
# app.py
import streamlit as st
import pandas as pd
import json
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

@st.cache_data
def load_and_train():
    data = []
    with open('News_Category_Dataset_v3.json', 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    
    categories = ['TECHNO', 'ENTERTAINMENT', 'POLITICS', 'BUSINESS']
    df = df[df['category'].isin(categories)].reset_index(drop=True)
    
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    
    def preprocess(text):
        doc = nlp(text.lower())
        tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
        return ' '.join(tokens)
        
    df['clean'] = df['headline'].apply(preprocess)
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(df['clean'])
    y = df['category']
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    return model, vectorizer, nlp

model, vectorizer, nlp = load_and_train()

def preprocess_for_app(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    return ' '.join(tokens)

st.title("NewsBot - Headline Classifier")
headline = st.text_input("Enter a news headline:")

if st.button("Predict Category"):
    if headline.strip():
        clean = preprocess_for_app(headline)
        vector = vectorizer.transform([clean])
        prediction = model.predict(vector)[0]
        st.success(f"Predicted Category: **{prediction}**")
```
