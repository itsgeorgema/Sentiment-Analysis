#impport and download necessary packages
import pandas as pd
import string
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
nltk.download("stopwords")

#load dataset
original_avengers_review = pd.read_csv("avengers_infinity_war_large_reviews.csv")
avengers_review=original_avengers_review.get(["Reviewer Name", "Rating", "Review Text"])
avengers_review.set_index("Reviewer Name")

#make a function to clean the data for the reviews
def preprocess_text(text):
    #remove punctuation and make lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    #remove stopwords by tokenizing, then reconstruct review with tokens to be a cleaned sentence fit for analysis
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

#create a row in avengers_review with cleaned data for the reviews
avengers_review["Cleaned_Review"] = avengers_review["Review Text"].apply(preprocess_text)


#function that analyzes the sentiment behind cleaned reviews
#polarity is part of textblob; you use textblob to analyze sentiment based on its polarity (what connotative end does it lean towards)
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

#implement sentiment analysis algorithm on dataframe and create a new column
avengers_review["Sentiment"] = avengers_review["Cleaned_Review"].apply(analyze_sentiment)

#show only the review and the sentiment behind it
avengers_review=avengers_review.drop(columns="Cleaned_Review")
print(avengers_review)

#query the dataset that was analyzed to determine how many rows have accurate sentiments that match the sentiment of the og dataset
avengers_review=avengers_review[avengers_review.get("Sentiment")==original_avengers_review.get("Sentiment")]
#determine the accuracy of the sentiment analysis algorithm through proportion of correct sentiments
print("The analysis was", str(avengers_review.shape[0]/original_avengers_review.shape[0]*100) + "%", "accurate")
