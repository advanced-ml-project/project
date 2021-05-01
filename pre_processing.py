"""
Useful functions to pre process text
"""
import unicodedata
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import word_tokenize

def remove_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)

def remove_numbers(text):
    '''Removes integers'''
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def remove_punctuation(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  
    return text

def text_lowercase(text):
    return text.lower()

def strip_whitespace(text):
    text = " ".join(text.split())
    return text

def remove_stopwords(text, lan='spanish', add_sw = []):
    if lan == 'spanish':
        sp = set(stopwords.words('spanish')).union(set(add_sw))
        words_tokens = word_tokenize(text)
        text = [i for i in words_tokens if i not in sp]
        text = ' '.join(text)

    elif lan == 'english':
        sp_en = set(stopwords.words('english'))
        words_tokens = word_tokenize(text)
        text = [i for i in words_tokens if i not in sp_en]
        text = ' '.join(text)

    else:
        print('''Could not remove stopwords. Please specify a language. 
        Only available in English or Spanish''')
        return

    return text

def apply_stemming(text, lan):
    if lan == 'spanish':
        stemmer = SnowballStemmer('spanish')
        words_tokens = word_tokenize(text)
        text = [stemmer.stem(i) for i in words_tokens]
        text = ' '.join(text)

    elif lan == 'english':
        stemmer_en = SnowballStemmer('english')
        words_tokens = word_tokenize(text)
        text = [stemmer_en.stem(i) for i in words_tokens]
        text = ' '.join(text)
    else:
        print('Stemmer only available in english or spanish')
        return
    
    return text

def clean_text(text, accents = True, punctuation=True, numbers=True, lowercase=True, 
          whitespace=True, lan = 'english', stopwords = False, stemming = False, add_sw=[]):

    if stopwords:
        text = remove_stopwords(text, lan = lan, add_sw=add_sw)
    
    if accents:
        text = remove_accents(text)

    if punctuation:
        text = remove_punctuation(text)

    if numbers:
        text = remove_numbers(text)

    if lowercase:
        text = text_lowercase(text)

    if whitespace:
        text = strip_whitespace(text)

    if stemming:
        text = apply_stemming(text, lan = lan)

    return(text)














