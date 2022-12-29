import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load the dataset
df = pd.read_csv("events.csv")

# Load the list of categories
categories = pd.read_csv("categories.csv")

# Create a dictionary to map category IDs to names
category_names = {row['id']: row['name'] for _, row in categories.iterrows()}

# Split the dataset into training and test sets
train_data = df[:int(len(df) * 0.8)]
test_data = df[int(len(df) * 0.8):]

# Extract the event names and categories from the training data
X_train = train_data['name']
y_train = train_data['category']

# Extract the event names from the test data
X_test = test_data['name']

# Create a TfidfVectorizer to convert the event names into numerical features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a LinearSVC classifier on the training data
clf = LinearSVC()
clf.fit(X_train_vec, y_train)

# Use the trained classifier to predict the categories of the test events
predictions = clf.predict(X_test_vec)

# Print the accuracy of the predictions
# print(f'Accuracy: {clf.score(X_test_vec, test_data["category"])}')

# Choose an event to classify
test_events = [
    "Meeting with my boss",
    "dinner with mom",
    "train in Gym"
]

for event in test_events:
    # Convert the event name into numerical features using the TfidfVectorizer
    event_vec = vectorizer.transform([event])

    # Use the classifier to predict the category of the event
    prediction = clf.predict(event_vec)

    # Print the prediction
    print(f'Predicted category for event "{event}": {category_names[prediction[0]]}')
