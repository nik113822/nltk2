import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re #regular expressions
from nltk import pos_tag
from nltk import word_tokenize
import preprocessor as p


f=open("python.txt")
text=f.read()
print(text)
p.set_options(p.OPT.MENTION)
clean = p.clean(text)
print(clean)




p.set_options(p.OPT.URL)  # καθάρισε μόνο URLs
clean = p.clean(text)
print(clean)
