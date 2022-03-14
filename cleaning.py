import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

cachedStopWords = stopwords.words("english")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
tokenizer = RegexpTokenizer('[\w+.]+')
st = nltk.PorterStemmer()
lm = nltk.WordNetLemmatizer()

def remove_contraction(text):
    '''remove contraction word'''
    # specific rule
    text = re.sub('won\'t', 'will not', text)
    text = re.sub('can\'t', 'can not', text)

    # general rule
    text = re.sub('n\'t', ' not', text)
    text = re.sub('\'re', ' are', text)
    text = re.sub('\'s', ' is', text)
    text = re.sub('\'d', ' would', text)
    text = re.sub('\'ll', ' will', text)
    text = re.sub('\'t', ' not', text)
    text = re.sub('\'ve', ' have', text)
    text = re.sub('\'m', ' am', text)
    return text

def remove_mentions(text):
    '''remove mentions'''
    return re.sub('@[a-zA-Z0-9_]+', '', text)

def remove_hashtags(text):
    '''remove hastags'''
    return re.sub('#[A-Za-z0-9_]+', '', text)

def remove_repeating_character(text):
    '''remove repeating character'''
    return re.sub(r'(.)\1{2,}', r'\1', text)

def remove_URLs(text):
    '''remove URLs'''
    return re.sub('(www.\S+)|(https?://\S+)', '', text)

def remove_punctuations(text):
    '''remove punctuations'''
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    '''remove stopword'''
    return ' '.join([word for word in word_tokenize(text) if word not in cachedStopWords])

def remove_numbers(text):
    '''remove numbers'''
    return re.sub('[0-9]+', '', text)

def tokenize(text):
    '''tokenize the input'''
    return tokenizer.tokenize(text)

def stemming(text):
    '''stemming the input'''
    return [st.stem(word) for word in text]

def lemmatizing(text):
    '''lemmatize the input'''
    return [lm.lemmatize(word, 'v') for word in text]

def join_text(text):
    '''combine list of string into string'''
    return ' '.join(map(str, text))

def clean_text(text):
    '''helper method to apply all the cleaning function'''
    text = text.lower()
    text = remove_contraction(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_repeating_character(text)
    text = remove_URLs(text)
    text = remove_punctuations(text)
    text = remove_stopwords(text)
    text = remove_numbers(text)
    text = tokenize(text)
    text = stemming(text)
    return text