# Feature-Ratings-Generator
## Abstract
This paper proposes a novel method to generate ratings from reviews using a Bayesian technique. One reason for the growing trend of online shopping in e-commerce platforms is its transparent review system, where a customer can review and rate a product that becomes open for others to see. Often, in making a purchase decision, a customer reads reviews to get feature-specific information about a product. These reviews, however, are becoming increasingly incomprehensible because of the large volume. Reading a sample of reviews may create a biased opinion as they do not represent overall reviews. To solve this problem, this project developed a fine-grained, feature-specific rating of a product from the reviews of customers using Bayesian estimation. This task is performed in three steps: (1) mining product features from the reviews of customers, (2) identifying the sentiment of the reviews that describe product features, (3) generating feature-specific ratings using Bayesian estimation.

 ## Methodology
The overall methodology implements a conceptual framework that consists of 4 sequential steps as in the figure below.

### Data collection and cleaning
A freely available Amazon product review dataset is collected from Kaggle. There were a total of 1599 reviews for 62 different products. The features such as reviews, and product were extracted. Any data point that had empty or null value in reviews and product column were filtered out as part of the data cleaning.

### Popular features extraction
The feature extraction process was completed in 2 steps. First, the potential features were extracted from the reviews. Then, the popular features were identified, and non- features were manually filtered out.

To generate the potential features, each customer review was broken into sentences using a sentence tokenizer of nltk library in Python. These sentences were stored in a separate sentence database. Then, each of these sentences were passed through Part-Of- Speech (POS) tokenizer, which parses each word in the sentence and identifies what part of speech it belongs to. The features are usually the nouns. Therefore, all the nouns and noun phrases were extracted from these sentences. These extracted nouns and noun phrases are the potential features.

Next, the goal was to identify popular features. The most frequently used nouns were considered to be the popular features. A transaction file was created, where each transaction contained nouns from a review sentence. Then, the Apriori algorithm3 was implemented with the support of 1%, which means if a noun appears in 1% of the rows in transaction file, then it is considered to be a popular feature. Then, from the list of top 100 popular nouns and noun phrases, the non-features were filtered out manually. This gave us a list of popular features that people have talked about in the reviews. One caveat with this approach of feature extraction is that it only considers those features that have been explicitly mentioned but doesn’t capture the implied features. For example: “The device was small” talks about the size of the product. However, since it doesn’t explicitly mention the word “size”, this sentence gets ignored during feature extraction.

### Sentiment Analysis

Then, each of the review sentence from the above step was classified into positive, negative, or neutral using a rule-based sentiment analyzer4. The neutral sentences do not contain strong opinion about the feature. Therefore, the neutral sentences were removed from the sentence database created above. At this stage, the sentence database contained the product names, review sentences, a list of popular features in that review sentence, and the sentiment of the review sentence as positive or negative.

### Rating generation
The next task was to generate ratings for each popular feature of a product. For
each product, all the review sentences that talked about at least one of the popular features were filtered. Then, for each of those features, a list of positive and negative sentences was created based on the observation in the filtered output. These observations of positive and negative sentences were used to generate ratings.
![Problem Statement](./Problem%20definition.png)

#### Frequentist Approach
Frequentist way of computing the rating is to average over all the observations. A positive sentence observation was considered to be of rating 5 and that of a negative sentence observation rating 1. Then, the expected value was computed with each of the observation being equally likely. Mathematically,
![Frequentist Approach](./Frequentist%20Approach.png)

#### Bayesian Approach
According to the Bayesian approach, a prior belief about the ratings of a feature of a product is stated. This belief is represented as a uniform prior distribution, where each of the ratings are equally likely. This represents a customer’s state of mind when he/she has no other information about the rating of a feature of a product. Then, a likelihood distribution is created that tells how likely it is to observe a positive or negative review sentence given the rating. Using these priors and likelihood, a posterior distribution of rating is generated based on the observation. Then, the posterior of the previous observation is used as the prior for the next observation. This way, an iterative algorithm will generate the posterior distribution having observed all the observations. Here, each of the positive or negative sentence observation is assumed to be conditionally independent of each other given the ratings. This assumption makes an intuitive sense because the likelihood of an observation doesn’t change given the rating is known. Mathematically,
![Bayesian Approach](./Bayesian%20Formula.png)
