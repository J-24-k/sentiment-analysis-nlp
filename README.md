
# Sentimental Analysis using NLP

End-to-end Sentiment Analysis project using NLP techniques and Random Forest on Amazon Alexa reviews.

This project applies Natural Language Processing and machine learning to classify Amazon Alexa reviews as positive or negative. Using tools like NLTK, CountVectorizer, and a Random Forest classifier, the project walks through a complete end-to-end sentiment analysis pipeline from text preprocessing to visualization and evaluation.

---

## Example Output

### Confusion Matrix  
Visualizing how well the model performs on test data:[`confusion_matrix.png`](confusion_matrix.png)

---

## Features

-  Cleaned and preprocessed review text using **NLTK**
-  Used **CountVectorizer** for feature extraction
-  Trained a **Random Forest Classifier** to classify sentiment
-  Evaluated the model using accuracy and confusion matrix
-  Visualized rating distribution,feedback types and variation trends

---


## How to Run the Code

1.Clone this repo:
```bash
git clone https://github.com/J-24-k/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your dataset:
Add `amazon_alexa.csv` to the `data/` folder

4. Run the script:
```bash
python "Sentimental_Analysis using NLP.py"
```

---

## Sample Output (Text-Based)

```plaintext
Training Accuracy: 0.99
Testing Accuracy: 0.94
```

---

## Example Prediction

```
Original review: "Alexa understands my commands clearly and responds quickly."
Processed: "alexa understands commands clearly responds quickly"
Prediction: Positive 👍
```
