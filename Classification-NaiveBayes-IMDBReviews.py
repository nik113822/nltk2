


"""
   Builds a Naive Bayes classifier for text data.

   This script builds a Naive Bayes classifier to predict the sentiment (positive/
   negative) of movie reviews in english. Builds a Naive Bayes using textual data.
   Uses the IMDB Movie Reviews dataset [1].
   
   NOTE: Due to its size, a smaller subset of the dataset is available (IMDBDataset-small.csv) 
         containing a random subset of the IMDB Movie Reviews. Contains approximately 7K reviews.
         This smaller set is available for supporting educational/explorational tasks.

   References:
      1) IMDB Dataset of 50K Movie Reviews, https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/


   Required modules:
      * See file requirements.txt 




   v1.0/mmt/Oct 2022
"""


import sys

import argparse
import webbrowser
import re # for regular expressions
import random # For some support tasks
from datetime import datetime # For debugging

import numpy as np 
import pandas as pd 


# NLTK module.
# Since the data processed by script to build
# the classifier is text (not numbers, strings etc) special
# text handling functions are needed. Such functions are available
# in the NLTK module
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Since a Bag of Words model will be used, the necessary
# function is included.
# See https://en.wikipedia.org/wiki/Bag-of-words_model
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score,precision_score,recall_score, precision_recall_fscore_support






# Data files (large and small version)
IMDB_DATA_SET_FILE       = 'IMDBDataset.csv'
IMDB_DATA_SET_FILE_SMALL = 'IMDBDataset-small.csv'

# Maximum number of features (words) to allow in
# count vectorizer
MAX_FEATURES = 1000

# Number of random samples to
# generate to watch the preprocessing
# steps. Used when debugging.
NUM_RANDOM_SAMPLES = 3

# Percentage (%) of dataset to use as
# the testing set. Allowed valies in the range (0, 1)
TEST_SET_SIZE_PCT = 0.3






###########################################################
#
# First we define some functions to make our life easier.
# NOTE: these will be used mainly for preprocessing
#
###########################################################



#
# We clean up any HTML element that may lurk in our data
#
def RemoveHTMLElements(text):
    """Removes elements that look like html elements i.e.
       text between < > tags.

    Parameters:   
       text -- string representing text
       return -- str
    """
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)


# Remove words that are contain symbols that are not
# letters or numbers.
#
def RemoveSpecialCharacters(text):
    """Removes characters that is not text (like , . - " etc.)

    Parameters:
       text -- string representing text
       return -- string without special characters
    """
    rem = ''
    for i in text:
        if i.isalnum()==False and i != " ":
            text=text.replace(i,'')
            
    return text



# Self explaining...
def ToLowerCase(text):
    """Makes text lowercase

    Parameters:
       text -- string representing text
       return -- string in lowercase
    """
    return text.lower()



#
# Remove stopwords (i.e. words like the, as, if, that etc)
# We use stopwords for english available by the nltk package (that's really cool!)
#
def RemoveStopWords(text, lang="english"):
    """Removes stop words from text.

    Parameters:
       text -- string representing text
       lang -- language of text. Defaults to english
       return -- string without language-specific stopwords
    """
    stop_words = set(stopwords.words(lang))
    words = word_tokenize(text)
    return  " ".join([w for w in words if w not in stop_words])



# We use the SnowballStemmer to stem words.
# Stemming means getting the word stem of a word which does not have any inflectional part.
# i.e. cats will become cat (word stem)
# enemies will become enem (word stem)
# baby will become bab
# etc
#
# We do this in order to not recognize inflectional morphs of a word as a different word.
# 
#
def WordStemming(text):
    """Reduces all inflected words found in text to their word stem.

    Parameters:
       text -- string representing text
       return -- string containing only word stems
    """
    ss = SnowballStemmer('english')
    #return " ".join([ss.stem(w) for w in text])
    return(" ".join([ss.stem(w) for w in text.split()]))






def main():

    #
    # Parse command line arguments, if there are any configuring the way Naive Bayes
    # should be executed.
    #

    clArgs = argparse.ArgumentParser(description='Command line arguments for Batch Gradient Descent')
    clArgs.add_argument('-I', '--info', action='store_true', help="Display general information about what the script does.") # Display only information
    clArgs.add_argument('-D', '--dataset',   action='store_true', help='Navigate to webpage containing dataset. Has also description of features.')
    clArgs.add_argument('-V', '--documenttermmatrix',   action='store_true', help='Display the count vectorizer resulting from text. For large datasets this might take some time.')
    clArgs.add_argument('-G', '--debug',   action='store_true', help='Enable debug mode.')
    arguments = vars( clArgs.parse_args() ) # Convert namespace to dictionary



    # Prelude/intro

    print("")
    print("")
    print('*'*92)
    print('*')
    print('* Sentiment analysis using Naive Bayes.')
    print('*')
    print('* Dataset: "IMDB Dataset of 50K Movie Reviews" available from ')
    print('*           https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews')
    print('*')
    print('*           File IMDBDataset-small.csv is a small subset of IMDBDataset.csv generated locally by randomly sampling ')
    print('*           rows. Used exclusively for debugging/testing purposes.')
    print('*')
    print('*'*92)
    print("")



    if arguments['info']:
       sys.exit(0)

    if arguments['dataset']:
       webbrowser.open('https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews')
       sys.exit(0)
   
    
    print("Reading dataset from file IMDBDataset-small.csv....", end="")
    data = pd.read_csv(IMDB_DATA_SET_FILE_SMALL, header=0)
    print("done.")


    
    #########################################################################################################
    # Descriptive statistics of data
    #########################################################################################################
    
    print("\tNumber of rows:", data.shape[0])
    print("\tNumber of features/columns:", data.shape[1])
    
    # Sentiment will be the class attribute.
    # Displayig the distinct values of class attribute and their count.
    print(data['sentiment'].value_counts())
    
    




    #########################################################################################################
    # Preprocessing
    #########################################################################################################


    print("Preprocessing...")

    # Python and modules like pandas/sklearn etc  does not have Factor datatype as R does.  
    # Also, strings cannot are not supported as nominal values. integers though can be used to represent
    # nominal values.
    # Hence, transform string sentiment values in class attribute to integers.
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1 })
  
    print(data.head(10))
    

    # Processing text.

    evalSamples = random.sample(range(0, data.shape[0]), NUM_RANDOM_SAMPLES)
    
    print("\tLower case....", end="")
    data['review'] = data['review'].apply(ToLowerCase)
    print("done.")

    # Some messages are only printed out if the -G option is set on
    # the command line that enables debug messages.
    if arguments['debug']:
       print(f'\n[DEBUG] {datetime.now().strftime("%d/%m/%Y@%H:%M:%S")}\n', data.iloc[evalSamples,:])

       
    print("\tCleaning html entities....", end="")
    data['review'] = data['review'].apply(RemoveHTMLElements)
    print("done")

    if arguments['debug']:
       print(f'\n[DEBUG] {datetime.now().strftime("%d/%m/%Y@%H:%M:%S")}\n', data.iloc[evalSamples,:])
    
    print("\tRemoving special characters....", end="")
    data['review'] = data['review'].apply(RemoveSpecialCharacters)
    print( "done.")

    if arguments['debug']:
       print(f'\n[DEBUG] {datetime.now().strftime("%d/%m/%Y@%H:%M:%S")}\n', data.iloc[evalSamples,:])

    print("\tRemoving stopwords....", end="")
    data['review'] = data['review'].apply(RemoveStopWords)
    print( "done" )

    if arguments['debug']:
       print(f'\n[DEBUG] {datetime.now().strftime("%d/%m/%Y@%H:%M:%S")}\n', data.iloc[evalSamples,:])

    print("\tStemming....", end="")
    data['review'] = data['review'].apply(WordStemming)
    print("done")

    if arguments['debug']:
       print(f'\n[DEBUG] {datetime.now().strftime("%d/%m/%Y@%H:%M:%S")}\n', data.iloc[evalSamples,:], '\n')





    
    # Here we create the CountVectorizer.
    # This means that we will create a huge matrix with individual words encounterred as columns
    # and as rows, the movie reviews. Each position in that matrix tells us how many times that word
    # appears in that review.
    # For example the CountVectorizer will transform the data set of two sentences "like holidays", "dislike lentils"
    # into a Document Term Matrix which looks like this:
    #
    #                       like | holidays | dislike | lentils
    #  "like holidays"       1        1          0         0
    #  "dislike lentils"     0        0          1         1  
    #
    # (numbers represent counts; how many times each word appears in the sentence) Now, both sentences have
    # been vectorized. Each individual word is a feature/column.
    
    print("Creating count vectorizer....", end="")
    
    cv = CountVectorizer(max_features = MAX_FEATURES)
    # Creating the Document Term Matrix.
    # See https://en.wikipedia.org/wiki/Document-term_matrix
    documentTermMatrix = cv.fit_transform(data['review']).toarray()

    # Transforming document term matrix which is represented as an
    # array into a data frame. This is done in order to avoid 
    # dealing with different data types which may complicate things.
    # Everything is dealt with in terms of a data frame. Since this
    # script serves educational purposes, primary concern is understanding
    # the solution, not optimization issues in terms of speed and memory.
    #
    # cv.get_feature_names_out() will return the actual words which
    # serve as features in the document term matrix.
    vectorizedReviewData = pd.DataFrame(data=documentTermMatrix, columns=cv.get_feature_names_out())
    print("done")




    
    # Show how the vectorized review data actually looks like.
    # We do this for educational purposes ONLY!
    # Columns: individual tokens/words
    # Rows: documents/text in the dataset
    # Cells in data frames: how many times word appears in doument/text
    #
    # NOTE: To display the vectorized review data add the -V argument on the command line.
    #       IMPORTANT! Displaying the vectorized review makes sense only for small number if reviews.
    #                  activating it for large number of review may hang your machine. You have been
    #                  warned.

    if arguments['documenttermmatrix']:
       print("")
       print("===============================================================================")
       print("How the vectorized matrix looks like:")
       print("")
       print(vectorizedReviewData)
       print("=============================================================================")
    


    # Prepare training and testing sets
    print('Generating training and testing set....', end="")
    trainingSetFeatures, testingSetFeatures, trainingSetSentiment, testingSetSentiment = train_test_split(vectorizedReviewData, data['sentiment'], test_size=TEST_SET_SIZE_PCT, random_state=9)
    print('done. (training set:', trainingSetFeatures.shape[0], 'rows testing set:', testingSetFeatures.shape[0], ')')








    #########################################################################################################
    # Training Naive Bayes model
    #########################################################################################################


    print("Training multinomial model....", end="")
    #
    # Train the Naive Bayes model.
    # NOTE: we use multinomialNB() because this is the most appropriate one
    #       when the data used is of type counts. And we do have counts here (see CountVectorizer).
    #       multinomialNB() refers to the distribution of the independent variables. MultinomialNB()
    #       assumes multinomial distribution of our variables.
    # alpha: smoothing factor (i.e. how to change counts when probability is 0, which in Naive Bayes
    #        is an issue.
    # fit_prior: should prior probabilities be calculated or not.
    mnb = MultinomialNB(alpha=1.0, fit_prior=True)

    # Train Naive Bayes model
    mnb.fit(trainingSetFeatures, trainingSetSentiment)
    print("done.")

    




    

    #########################################################################################################
    # Predictions on the test set
    #########################################################################################################
    
    print("\n\nPredicting class (sentiment) on the testing set.")

    # Use the testing set to do some predictions.
    testingSetPredictions = mnb.predict(testingSetFeatures)





    #########################################################################################################
    # Model evaluation
    #########################################################################################################
    
    print('Testing set evaluation metrics:')
    
    # Calculate some metrics: accuracy, precision and recall for
    # the testing set
    #
    # IMPORTANT! Functions precision_score and recall_score return respectively the
    # precision and recall of one class only - the one designated with the label 1.
    # You may show these metrics for each class using precision_recall_fscore_support

    print("\tAccuracy:", accuracy_score(testingSetSentiment, testingSetPredictions))

    # IMPORTANT! for precision and recall we don't have to pass an argument average
    # since the class attribute of this dataset here, takes only 2 values and the default averaging method
    # for precision_score and recall_score is 'binary'.
    # Should the class attribute have more than 2 values, calling precision_score() and/or recall_score()
    # without average argument will result in an error. In these cases these functions should be called
    # with average having values micro, macro or weighted. See
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    print("\tPrecision:", precision_score(testingSetSentiment, testingSetPredictions))
    print("\tRecall=:", recall_score(testingSetSentiment, testingSetPredictions))






    #########################################################################################################
    # Predictions for UNKNOWN data/reviews
    #########################################################################################################
    
    print("\n\nPredicting class (sentiment) on unknown data.")
    
    # We have 2 reviews here for which we want to predict the sentiment
    unknownReviewsSource = [ "There is NO excuse for how bad this movie was. It's so out of touch with the world that it's actually offensive and not in a sexual like the title might lead you to believe.".lower(),
                       'A phenomenal achievement and a real candidate for the greatest motion picture ever made.'.lower()]

    
    print("Total or ", len(unknownReviewsSource), "unknown reviews. (NOTE: Predicted class uses recoded values of class variable)")


    # Do exactly the same preprocessing the training set went through
    # Note: make this an exercise....

    # First, make a copy since preprocessing will change the review content.
    unknownReviews = unknownReviewsSource.copy()
    
    unknownReviews = [RemoveSpecialCharacters(r) for r in unknownReviews]
    unknownReviews = [RemoveHTMLElements(r) for r in unknownReviews]
    unknownReviews = [RemoveStopWords(r) for r in unknownReviews]
    unknownReviews = [WordStemming(r) for r in unknownReviews]



    #
    # Transform the text into a vector, indicating which word appears in the text and how many times, using
    # the features (rows) of the trained Naive Bayes model.
    #

    # Using the CountVectorizer generated from the training set to vectorize unknown reviews.
    vectorizedUnknownReviews = cv.transform(unknownReviews)
    vectorizedUnknownReviewData = pd.DataFrame(vectorizedUnknownReviews.toarray(), columns=cv.get_feature_names_out())
    
   
    
    # Ok done. Use the trained Naive Bayes to predict sentiment
    predictedUnknownSentiment = mnb.predict(vectorizedUnknownReviewData)    

    # Use the model to predict sentiment for reviews.
    # NOTE: we display also the calculated probabilities for each
    #      category/class (i.e. positive/negative).
    #      The .predict_proba() returns also class probabilities that
    #      .predict() does not.
    unknownReviewPredictions = mnb.predict_proba(vectorizedUnknownReviewData)
    #idx = 0 # counting rows to be able to refer to specific review. 
    for i, prb in enumerate(unknownReviewPredictions):
       # find position of maximum probability
       pos = list(prb).index( max(prb) )    
       print("\t* Unknown review at ", str(i), ")", sep='')
       print("\t\tReview:", unknownReviewsSource[i])
       print("\t\tProbabilities:", prb) 
       # mnb.classes_ is a list of all encounterred classes in the training set
       print("\t\tpredicted class = ", mnb.classes_[pos] )
       




# Main guard
# See: https://stackoverflow.com/questions/19578308/what-is-the-benefit-of-using-main-method-in-python
#
# Script starts execution from here.
if __name__ == "__main__":
    main()


