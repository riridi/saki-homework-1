# Exercise 1 - Classification
# Task: For a given set of financial transactions, 
# classify each one into one of seven revenue (transaction) categories; the categories are:
# income, private, living, standard of living, finance, traffic, leisure

# imported packages
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

# read the csv file
transactions = pd.read_csv('SAKI Exercise 1 - Transaction Classification - Data Set.csv', sep=';')

print("* Read CSV file infos:")
print("Shape: " + str(transactions.shape))
print("Columns: " +  str(list(transactions.columns)) + "\n")

# Clean the dataset: remove unnessesary columns
print("* Removing columns: ID, Waehrung, Valutadatum \n")
transactions.drop('Unnamed: 0', 1, inplace=True)  # remove id
transactions.drop('Waehrung', 1, inplace=True)  # remove Waehrung (always the same)
transactions.drop('Valutadatum', 1, inplace=True)  # remove Valutadatum (no feature)

print("* Table after removing unwanted columns:")
print("Shape: " + str(transactions.shape))
print("Columns: " +  str(list(transactions.columns)) + "\n")

# Prepare the dataset
print("* Prepare dataset by modifying/adding features")
# Fill up missing entries in Auftragskonto
transactions['Auftragskonto'].fillna("0", inplace=True)

# Change Buchungstag to weekdays
print("Changing Buchungstag to weekdays")
transactions['Buchungstag'] = transactions['Buchungstag'].map(lambda date: str(datetime.strptime(date, '%d.%m.%Y').weekday()))

# split Verwendungszweck in different features
usage = pd.DataFrame(transactions['Verwendungszweck'].map(lambda line: line.split(' ')))  # split words

# Helper function to extract features
def find_value_for_key(array, key):
    try:
        return array[array.index(key) + 1]
    except (ValueError, IndexError):
        return "None"

# Helper function to remove extracted features
def remove_key_and_value(array, key):
    try:
        array.remove(find_value_for_key(array, key))
        array.remove(key)
        return array
    except (ValueError, IndexError):
        return array

print("Adding additional features Kundenreferenz and End-To-End-Ref from Verwendungszweck")
# additional feature Kundenreferenz
transactions['Kundenreferenz'] = usage['Verwendungszweck'].map(lambda arr: find_value_for_key(arr, 'Kundenreferenz:'))

# additional feature End-To-End-Ref.
transactions['End-To-End-Ref'] = usage['Verwendungszweck'].map(lambda arr: find_value_for_key(arr, 'End-To-End-Ref.:'))

# remove Kundenref + End to End
usage['Verwendungszweck'] = usage['Verwendungszweck'].map(lambda arr: remove_key_and_value(arr, 'Kundenreferenz:'))
usage['Verwendungszweck'] = usage['Verwendungszweck'].map(lambda arr: ' '.join(remove_key_and_value(arr, 'End-To-End-Ref.:')))

# Use countvectorizer on the rest of the words in Verwendungszweck
print("Extract additional features from Verwendungszweck through CountVectorizer")
vectorizer = CountVectorizer()
messages = usage['Verwendungszweck'].values
counts = vectorizer.fit_transform(messages)
counts_df = pd.DataFrame(counts.todense(), columns=vectorizer.get_feature_names())

# Concatinate counts with transactions
transactions = pd.concat([transactions, counts_df], axis=1)

transactions.drop('Verwendungszweck', 1, inplace=True)  # remove Verwendungszweck
print("New shape of transactions: " + str(transactions.shape) + "\n")

# Split in Training and Test Set
data_randomized = transactions.sample(frac=1, random_state=1)

# Split features and labels
feat_col = data_randomized.columns.drop(['label'])
features = pd.get_dummies(data_randomized[feat_col])
labels = pd.DataFrame(data_randomized['label'])

print("* Split dataset into features and labels:")
print("Feature set: " + str(features.shape))
print("Lable set: " + str(labels.shape) + "\n")

# Calculate index for split
split_index = round(len(data_randomized) * 0.8)

# Split into training and test sets
training_features = features[:split_index].reset_index(drop=True)
test_features = features[split_index:].reset_index(drop=True)

training_labels = labels[:split_index].reset_index(drop=True)
test_labels = labels[split_index:].reset_index(drop=True)

print("* Split dataset into training and test set:")
print("Training features: " + str(training_features.shape))
print("Training labels: " + str(training_labels.shape))
print("Test features: " + str(test_features.shape))
print("Test labels: " + str(test_labels.shape) + "\n")

# Train model
# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
print("* Train model \n")
model.fit(training_features, training_labels.values.ravel())

# Validate model
predictions = model.predict(test_features)

print("* Validate model - Scores:")
print("Accuracy = " + str(accuracy_score(test_labels, predictions)))
print("Precision = " + str(precision_score(test_labels, predictions, average="weighted")))
print("Recall = " + str(recall_score(test_labels, predictions, average="weighted")))
print("F1 = " + str(f1_score(test_labels, predictions, average="weighted")) + "\n")

# 5-fold Cross validation
print("* Cross validation: ")

scores = cross_val_score(model, features, labels.values.ravel(), cv=5)

print("Accuracy of the 5 folds: " + str(scores))
print("Mean accuracy: " + str(scores.mean()) + "\n")
