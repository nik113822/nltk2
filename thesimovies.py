import nltk
from nltk.corpus import movie_reviews

# Αν λείπει:
# nltk.download('movie_reviews')

# λίστα όλων των αρχείων (με υποφακέλους)
print(movie_reviews.fileids()[:10])

# πλήρες μονοπάτι ενός αρχείου
print(movie_reviews.abspath('neg/cv000_29416.txt'))

# ριζικός φάκελος του corpus
print(nltk.data.find('corpora/movie_reviews'))

from nltk import pos_tag
from nltk import word_tokenize

text = "NLTK is great for NLP!"
words = word_tokenize(text)
pos_tags = pos_tag(words)
print(pos_tags)


