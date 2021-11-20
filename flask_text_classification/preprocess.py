import re ## regular expression
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import spacy
import es_core_news_sm ## import the model 
import unidecode
import pickle

nlp = es_core_news_sm.load() ## load the model
sb_spanish = SnowballStemmer('spanish') ## quitar el plural
nltk.download("punkt") ## Sentence tokenizer | divide a text


#################################################################
######### PREPROCESSING
#################################################################

def regex(text):
    ## delete any character that not is a word
    new_text = re.sub("\W+", " ", text)    
    ## delete a number [+] one or more ocurrences
    new_text = re.sub("\d+", " ", new_text).strip()

    return new_text

def lower_case(text):
    ## put lowercase
    new_text = text.lower()
    
    return new_text

def tokens(text):
    ## tokenize words
    new_text = word_tokenize(text, language="spanish")
    
    return new_text

def stop_words(text):    
    ## read stopwords
    file = open("spanishST.txt", "r", encoding='latin-1')
    stop = file.read().splitlines()
    file.close()

    new_text = []    

    ## remove stop words
    for w in text:
        if w not in stop:
            new_text.append(w)
            
    return new_text     

def stemmer(text):
    new_text = []
    ## get the lema by ech token
    for t in text:
        new_text.append(sb_spanish.stem(t))
        
    return new_text

def lemma(text):
    new_text = []
    
    ## get lemmas    
    doc = nlp(" ".join(text))
    for tok in doc:
        new_text.append(tok.lemma_)
        
    return new_text

def join_all(text):
    ## join
    new_text = " ".join(text)
    
    return new_text

def unidecode_text(text):
    ## remove accents
    new_text = unidecode.unidecode(text)
    
    return new_text

#################################################################
######### NORMALIZATION
#################################################################

def preprocess(text):
    ## apply all preprocess
    
    text = regex(text)
    text = lower_case(text)
    text = tokens(text)
    text = stop_words(text)
    text = lemma(text)
    text = join_all(text)
    text = unidecode_text(text)    
    
    return text

#################################################################
######### TF-IDF
#################################################################
tf_idf = pickle.loads(open("tf_idf.pickle", "rb").read())
def tf_idf_vector(text):
    text_preprocessed = preprocess(text)
    text_tf_idf = tf_idf.transform([text_preprocessed])

    return text_tf_idf

#################################################################
######### SVM
#################################################################
svm = pickle.loads(open("svm_leadership.pkl", 'rb').read())
def svm_classifier(text):
    text_tf_idf = tf_idf_vector(text)
    print(text_tf_idf)
    prediction = svm.predict(text_tf_idf)

    return prediction

