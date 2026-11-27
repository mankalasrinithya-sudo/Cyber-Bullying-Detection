
import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# 1. Clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# 2. Load the Jigsaw dataset
data = pd.read_csv("dataset/jigsaw/train.csv")

print("Original dataset shape:", data.shape)
print("Columns:", data.columns.tolist())


# 3. Identify the six label columns
label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Check that the required columns exist
required_columns = ["comment_text"] + label_columns

for column in required_columns:
    if column not in data.columns:
        raise ValueError("Missing column: " + column)


# 4. Remove rows without text
data = data.dropna(subset=["comment_text"])

# Remove exact duplicate records
data = data.drop_duplicates()

# Combine duplicate comments and retain any positive toxicity label
data = data.groupby("comment_text", as_index=False)[label_columns].max()


# 5. Convert six labels into one binary label
data["label"] = (data[label_columns].max(axis=1) > 0).astype(int)

# Clean the comments
data["clean_text"] = data["comment_text"].apply(clean_text)

# Remove empty comments after cleaning
data = data[data["clean_text"] != ""]


# 6. Display class distribution
print("\nClass distribution:")
print(data["label"].value_counts())

print("\n0 = Not Cyberbullying")
print("1 = Cyberbullying")


# 7. Separate text and labels
X = data["clean_text"]
y = data["label"]


# 8. Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 9. Convert text into numerical TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 10. Create the three models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB()
}


# 11. Train and evaluate each model
results = []

for name, model in models.items():

    print("\n------------------------------")
    print("Training:", name)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not Cyberbullying", "Cyberbullying"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })


# 12. Save comparison results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))


# 13. Save all three trained models and the vectorizer
model_data = {
    "vectorizer": vectorizer,
    "models": models
}

joblib.dump(model_data, "models/all_models.pkl")

print("\nAll models saved successfully!")
