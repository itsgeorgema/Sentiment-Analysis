#import and download necessary packages
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
nltk.download('punkt_tab')
nltk.download("stopwords")

#load dataset
original_avengers_review = pd.read_csv("avengers_infinity_war_large_reviews.csv")
avengers_review=original_avengers_review.get(["Reviewer Name", "Rating", "Review Text"])
avengers_review.set_index("Reviewer Name")


#function that analyzes the sentiment reviews based on the average polarity of tokens
#polarity is part of textblob; you use textblob to analyze sentiment based on polarity (what connotative end does it lean towards)
def analyze_sentiment(text):
    #clean the text and split it into tokens
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)
    #remove stopwords from lists of tokens
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    #calculate polarity for each token and set the corresponding token in the list to the polarity of the token
    token_polarities = [TextBlob(token).sentiment.polarity for token in tokens]
    #determine sentiment based on the average polarity of all the tokens
    if not token_polarities:  # Handle case where all tokens are stopwords
        return "Neutral"
    if sum(token_polarities) / len(token_polarities) > 0:
        return "Positive"
    elif sum(token_polarities) / len(token_polarities) < 0:
        return "Negative"
    else:
        return "Neutral"

#implement sentiment analysis algorithm on dataframe and create a new column
avengers_review["Sentiment"] = avengers_review["Review Text"].apply(analyze_sentiment)

#show only the review and the sentiment behind it
print(avengers_review)

#query the dataset that was analyzed to determine how many rows have accurate sentiments that match the sentiment of the og dataset
avengers_review=avengers_review[avengers_review.get("Sentiment")==original_avengers_review.get("Sentiment")]
#determine the accuracy of the sentiment analysis algorithm through proportion of correct sentiments
print("The analysis was", str(avengers_review.shape[0]/original_avengers_review.shape[0]*100) + "%", "accurate")