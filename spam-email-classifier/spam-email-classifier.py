from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample dataset
emails = [
    "Win a free iPhone now",
    "Meeting scheduled tomorrow",
    "Congratulations you won money",
    "Please review the document",
    "Claim your free reward now"
]

labels = [1, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam

# ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(emails, labels)

# Test prediction
test_email = ["Free money waiting for you"]
prediction = model.predict(test_email)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")
