import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from multiprocessing import Pool
from apyori import apriori
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# extracts required columns and saves it in the 
def exportToExcel():
    df = pd.read_csv("original_dataset.csv")
    columns = ['name','reviews.date','reviews.numHelpful','reviews.rating','reviews.text','reviews.title','reviews.username']

    data = df[columns]
    data.to_excel("master_data_0827.xlsx")

#Loading the data
df = pd.read_excel("master_data_0827.xlsx")

# creating a product and product id pairs
product_name_id_df = pd.DataFrame(columns = ["product.id", "product.name"])
for i, name in enumerate(df.name.unique()):
    new_row = pd.Series({"product.id": i, "product.name": name})
    product_name_id_df = pd.concat([product_name_id_df, new_row.to_frame().T], ignore_index=True)

def get_product_id(name):
    return int(product_name_id_df.query("`product.name` == @name")["product.id"])

def get_product_name(i):
    return str(product_name_id_df.query("`product.id` == @i")["product.name"][i])

# create a product & sentence transactions dataframe
product_review_transaction_df = pd.DataFrame(columns = ["user.id", "review.sentence","review.ratings", "product.id", "nouns", "feature_nouns"])

for index, item in df.iterrows():
    sentences = sent_tokenize(item["reviews.text"])
    product_id = get_product_id(item["name"])
    review_rating = item["reviews.rating"]
    for sentence in sentences:
        # tokenize the reviews to the sentences
        row = pd.Series({"user.id": index, "review.sentence": sentence, "product.id": product_id, "review.ratings": review_rating})
        # add each sentence to the product_review_transaction dataframe
        product_review_transaction_df = pd.concat([product_review_transaction_df, row.to_frame().T], ignore_index=True)

# extract nouns from each transaction (review sentence)
stop_words = set(stopwords.words('english'))
wordnetLemmatizer = WordNetLemmatizer()
noun_filter = lambda e: e.startswith('NN')
adjective_filter = lambda e: e.startswith('JJ')

def tag(sen, filter_wrapper):
    wordsList = nltk.word_tokenize(sen)
    wordsList = [w for w in wordsList if not w in stop_words]
    tagged = nltk.pos_tag(wordsList)
    filtered_tagged = filter(lambda e: filter_wrapper(e[1]), tagged)
    stemmed_tagged = map(lambda w: wordnetLemmatizer.lemmatize(w[0]), filtered_tagged)
    lowercase_tagged = map(lambda w: w.lower(), stemmed_tagged)
    return list(lowercase_tagged)

noun_tokenized_reviews = list(map(lambda i: tag(i, noun_filter), list(product_review_transaction_df.iloc[:]['review.sentence'])))
# adjective_tokenized_reviews = list(map(lambda i: tag(i, adjective_filter), list(product_review_transaction_df.iloc[:]['review.sentence'])))
product_review_transaction_df["nouns"] = noun_tokenized_reviews
# product_review_transaction_df["adjectives"] = adjective_tokenized_reviews

# get all the most common nouns based on 1% support
# These nouns are the filtered candidate for the most common features
apriori_result = pd.DataFrame(apriori(product_review_transaction_df["nouns"], min_support=0.01))

pattern = re.compile("{'*'}")
all_features = apriori_result['items'].astype('string').str.removeprefix("frozenset({'").str.removesuffix("'})")[:80]
print("All selected features:", list(set(all_features)))

# hand pick the potential features from the frequent features
hand_picked_features = ['battery', 'camera', 'display', 'cord', 'echo', 'range', 'sound', 'search', 'content', 'time', 'access', 'quality']

for i, row in product_review_transaction_df.iterrows():
    row["feature_nouns"] = list(set(hand_picked_features).intersection(set(row["nouns"])))

total_reviews_with_features = list(filter(lambda e: e[1]["feature_nouns"], product_review_transaction_df.iterrows()))
total_review_sentence = len(product_review_transaction_df.iloc[:]["feature_nouns"])

temp_users = set()
x = list(map(lambda e: temp_users.add(e[1]["user.id"]), total_reviews_with_features))
users_with_features = len(temp_users)
total_users = df.shape[0]

print("Users with features: %d" % (users_with_features))
print("Total Users: %d" % (total_users))
print("Percent of users who reviewed common features: %.2f" % (users_with_features/total_users))

# sentiment analysis
analyzer = SentimentIntensityAnalyzer()
def vadersentimentanalysis(review):
    vs = analyzer.polarity_scores(review)
    return vs['compound']

def vader_analysis(compound):
    if compound >= 0.5:
        return 'positive'
    elif compound <= -0.5:
        return 'negative'
    
    return 'neutral'

# classify each sentence as either positive, negative, or neutral
product_review_transaction_df['review.sentiment'] = product_review_transaction_df['review.sentence'].apply(vadersentimentanalysis).apply(vader_analysis)

# filter out the neutral sentences
product_review_transaction_df[(product_review_transaction_df['review.sentiment'] != 'neutral') & (product_review_transaction_df['feature_nouns'].apply(lambda e: len(e) != 0))]

def annotater(x):
    if x == 'positive':
        return 'p'
    elif x == 'negative':
        return 'n'
    return 'o'

# get an object where each product maps to a list of features that contains a list of sentiments of review sentences
def get_feature_reviews():
    product_feature_reviews_map = {}
    all_product_ids = product_review_transaction_df['product.id'].unique()
    sentiment_filter = product_review_transaction_df['review.sentiment'] != 'neutral'
    for product_id in all_product_ids:
        product_filter = product_review_transaction_df['product.id'] == product_id
        product_feature_reviews_map[product_id] = {}
        for feature in hand_picked_features:
            feature_filter = product_review_transaction_df['feature_nouns'].apply(lambda e: feature in e)
            review_sentiments = product_review_transaction_df[sentiment_filter & product_filter & feature_filter]['review.sentiment']
            annotated_review_sentiments = list(review_sentiments.map(annotater))
            if annotated_review_sentiments:
                product_feature_reviews_map[product_id][feature] = annotated_review_sentiments
    
    return product_feature_reviews_map

product_feature_reviews_map = get_feature_reviews()


# Bayesian class that updates its prior on observation
class bayes_prob:
    def __init__( self, ratingsPrior, ratings, probPosReviewsGivenRating ):
        self.probRatings = ratingsPrior
        self.ratings = ratings
        self.probPosReviewsGivenRating = probPosReviewsGivenRating

    def probReviewGivenRating( self, observed, rating ):
        if observed == 'p':
            return self.probPositiveReviewGivenRating( rating )
        elif observed == 'n':
            return self.probNegativeReviewGivenRating( rating )
        else:
            return None

    # probability (review is positive | rating is *rating)
    def probPositiveReviewGivenRating( self, rating ):
        return self.probPosReviewsGivenRating[rating - 1]
    
    # probability (review is negative | rating is *rating)
    def probNegativeReviewGivenRating( self, rating ):
        return 1 - self.probPositiveReviewGivenRating( rating )
    
    # probability (rating is positive)
    def probPositivePostRatings( self ):
        return sum( self.probRatings * self.probPosReviewsGivenRating )
    
    # probability (rating is negative)
    def probNegativePostRatings( self ):
        return sum( (1 - self.probPosReviewsGivenRating) * self.probRatings )

    def getProbObservedRatings( self, observed ):
        if observed == 'p':
            return self.probPositivePostRatings()
        elif observed == 'n':
            return self.probNegativePostRatings()
        else:
            return None

    def getProbRatings( self, rating ):
        return self.probRatings[rating - 1]

    def getFinalRating(self):
        return sum(self.ratings * self.probRatings)


    def setProbRatings( self, observed ):
        total = self.getProbObservedRatings( observed )
        for rating in self.ratings:
            pre_prob = self.getProbRatings( rating )
            self.probRatings[rating - 1] = self.probReviewGivenRating( observed, rating ) * self.getProbRatings( rating ) / total
#             print("Observed: %c, m: %.2f, pre-prob: %.4f, post_prob: %.4f" % (observed, rating, pre_prob, a))

    def print_f( self ):
        for rating in self.ratings:
            print("p(rating = %.1f | data) = %.4f" % (rating, self.getProbRatings(rating)))
        print("p(review | data) = %.4f" % (self.probPositivePostRatings()))
    
# iterate over the product review dataset and compute rating using bayesian estimation
def bayes_estimation( product_feature_reviews_map ):
    product_feature_ratings = {}
    for product in product_feature_reviews_map:
        product_name = get_product_name(product)
        sorted_features = sorted(product_feature_reviews_map[product], key = lambda key: len(product_feature_reviews_map[product][key]), reverse = True)
        feature_ratings = dict()
        for feature in sorted_features:
            probRatings = np.ones(5, dtype='float') * 0.2
            ratings = np.arange(1,6,1)
            probPositiveReviewGivenRatings = np.arange(0.1,1.0,0.2)
            data = product_feature_reviews_map[product][feature]
            bayes = bayes_prob( probRatings, ratings, probPositiveReviewGivenRatings )

            for data_i in data:
                bayes.setProbRatings( data_i )

            bayes.print_f()
            feature_ratings[feature] = bayes.getFinalRating()
        product_feature_ratings[product_name] = feature_ratings
    return product_feature_ratings

result = bayes_estimation(product_feature_reviews_map)

print(result)
