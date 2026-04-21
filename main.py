
# Κατέβασε το μοντέλο "punkt"
import nltk
#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('stopwords')
nltk.download()



for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
            "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)
print("katevasthkan kapoia paketa pou thelame")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#Sentence tokenization
text = "Hello there! This is a short test. Cool, right?"
words = word_tokenize(text)
stop_words = set(stopwords.words("english"))
content_words = [w for w in words if w.isalpha() and w.lower() not in stop_words]
print(content_words)

