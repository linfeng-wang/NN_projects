#%%
import nltk
from ntkl.stem.porter import PorterStemmer
nltk.download('punkt')

stemmer = nltk.stem.PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    pass

