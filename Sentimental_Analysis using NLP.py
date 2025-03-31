
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

import nltk
from nltk.stem.porter import PorterStemmer 
nltk.download ('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
import pickle
import re
import os

# Load the data
data = pd.read_csv('data/amazon_alexa.csv')  # Changed from personal path to GitHub-safe relative path
data.head()
print(data.shape)

# Column names
print(f"Feature names: {data.columns.values}")

# Check for null values
data.isnull().sum()

# Creating a new column 'length' that will contain the length of the string in 'verified_reviews' column
data['length']= data['verified_reviews'].apply(len)
data.head()

# Randomly checking for 10th record
# Caution: Avoid printing raw text from datasets with potentially sensitive info
# print(f" 'verified_reviews' column value: {data.iloc[10]['verified_reviews']}") 
print(f" Length of review: {len(data.iloc[10]['verified_reviews'])}")  # Length of review using len()
print(f"'length' column value : {data.iloc[10]['length']}")  # Value of the column 'Length'

data.dtypes

# Analyzing 'rating' column
print(f"Rating value count:\n {data['rating'].value_counts()}")
data['rating'].value_counts().plot.bar(color = 'red')
plt.title('Rating distribution count')
plt.xlabel('Ratings') 
plt.ylabel('Count')
plt.show()

# Finding the percentage distribution of each rating
print(f"Rating value count - percentage distribution:\n{round(data['rating'].value_counts()/data.shape[0]*100,2)}")
fig = plt.figure(figsize=(7,7))
colors = ('red', 'green', 'blue', 'orange', 'yellow')
wp = {'linewidth':1, "edgecolor": 'black'}
tags = data['rating'].value_counts()/data.shape[0]
explode=(0.1,0.1,0.1,0.1,0.1)
tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode)

from io import BytesIO
graph = BytesIO()
fig.savefig(graph, format="png")
plt.title("Rating Percentage Distribution")  
plt.show()

# Distinct values of 'feedback' and its count
print(f"Feedback value count: \n{data['feedback'].value_counts()}")

# Caution: Avoid printing raw text from datasets with potentially sensitive info
# review_0 = data[data['feedback'] ==0].iloc[1]['verified_reviews']
# print(review_0)
# review_1 = data[data['feedback'] ==1].iloc[1]['verified_reviews']
# print(review_1)

data['feedback'].value_counts().plot.bar(color = 'blue')
plt.title('Feedback distribution count')
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.show()

# Feedback = 0
data[data['feedback'] == 0]['rating'].value_counts()
# Feedback = 1
data[data['feedback'] == 1]['rating'].value_counts()

# Distinct values of 'variation' and its count
print(f"Variation value count: \n{data['variation'].value_counts()}")
data['variation'].value_counts().plot.bar(color = 'orange')
plt.title('Variation distribution count')
plt.xlabel('Variation')
plt.ylabel('Count')
plt.show()

data.groupby('variation')['rating'].mean()
data.groupby('variation')['rating'].mean().sort_values().plot.bar(color = 'brown', figsize=(11, 6))
plt.title("Mean rating according to variation") 
plt.xlabel('Variation')
plt.ylabel('Mean rating')
plt.show()

# Analyzing 'verified_reviews' column
data['length'].describe()
data.groupby('length')['rating'].mean().plot.hist(color = 'blue', figsize=(7, 6), bins = 20)
plt.title("Review length wise mean ratings") 
plt.xlabel('ratings') 
plt.ylabel('length')
plt.show()

# Preprocessing and Modelling
corpus = []
stemmer = PorterStemmer()
for i in range (0, data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values

# Create the Models directory if it does not exist
if not os.path.exists('Models'):
    os.makedirs('Models')

# Saving the Count Vectorizer (not storing sensitive user data, only vector mapping)
pickle.dump(cv, open('Models/countVectorizer.pkl', 'wb'))

# Checking the shape of X and y
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")

# Splitting data into train and test set with 30% data with testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")

print(f"X train max value: {X_train.max()}")
print(f"X test max value: {X_test.max()}")

scaler = MinMaxScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

# Saved for reuse in model deployment — avoid saving data-specific state
pickle.dump(scaler, open('Models/scaler.pkl', 'wb'))

# Fitting scaled X_train and y_train on Random Forest Classifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)

# Accuracy of the model on training and testing data
print("Trainig Accuracy:", model_rf.score(X_train_scl, y_train))
print("Testing Accuracy:", model_rf.score(X_test_scl, y_test))

# Predicting on the test set
y_preds = model_rf.predict(X_test_scl)

# Confusion Matrix
cm = confusion_matrix(y_test, y_preds)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_rf.classes_)
cm_display.plot()
plt.show()
