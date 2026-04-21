from nltk.corpus import movie_reviews
import nltk
nltk.download('movie_reviews')   # κατεβάζει το corpus

# κατηγορίες
print(movie_reviews.categories())

#import nltk
import random
#from nltk.corpus import movie_reviews

documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

random.shuffle(documents)

all_words = []# Θα μαζέψει όλες τις λέξεις του corpus (σε πεζά)
for w in movie_reviews.words():# Παίρνει κάθε λέξη από ΟΛΟ το corpus
    all_words.append(w.lower())# την κατεβάζει σε lower και την προσθέτει

all_words = nltk.FreqDist(all_words)# Συχνότητες εμφάνισης για κάθε λέξη
word_features = list(all_words.keys())[:3000]# Κρατά τα 3000 πιο συχνά tokens ως features

def find_features(document): # Μετατρέπει ένα έγγραφο -> λεξικό features
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)# True/False αν υπάρχει στο έγγραφο
    return features

print(find_features(movie_reviews.words('neg/cv000_29416.txt')))
# Δείχνει το feature dict για ΜΙΑ συγκεκριμένη αρνητική κριτική
featuresets = [(find_features(rev), category) for (rev, category) in documents]
# Φτιάχνει λίστα από (features, ετικέτα) για ΟΛΕΣ τις κριτικές — έτοιμο για training/test