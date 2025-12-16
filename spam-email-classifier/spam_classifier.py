# spam_classifier.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample data (no CSV needed)
emails = [
    ("Congratulations! You won a free iPhone. Click here to claim.", "spam"),
    ("Reminder: Your appointment is scheduled for tomorrow.", "ham"),
    ("Get cheap loans now!!! Limited offer", "spam"),
    ("Hi John, can we reschedule the meeting?", "ham"),
    ("Win cash prizes every week, join now!", "spam"),
    ("Please review the attached report.", "ham")
]

# Split into text and labels
X, y = zip(*emails)

# Convert text to numbers
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Test with new email
new_email = ["Congratulations! You won a prize. Click here!"]
new_email_vec = vectorizer.transform(new_email)
prediction = model.predict(new_email_vec)
print("New email prediction:", prediction[0])
