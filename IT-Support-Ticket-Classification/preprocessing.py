import pandas as pd
import re
import spacy
import constant as ct

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('words')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = set(nltk.corpus.words.words())

nlp = spacy.load('en_core_web_sm')



def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in ct.Stopwords])

def get_lemm(text):
    return " ".join([lemmatizer.lemmatize(words) for words in text.split()])

def get_gt2(text):
    return " ".join([word for word in text.split() if len(word)>2])

def get_lemm_spacy(text):
    doc = nlp(text)
    return  " ".join([token.lemma_ for token in doc])

def clean(Text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', Text)  # remove punctuations
    #text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)  
    text = get_lemm_spacy(text)
    
    text = remove_stopwords(text)
    text = re.sub(r'\d+', '', text)
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w in words or not w.isalpha())
    text = get_gt2(text)

    return text

def get_data(file_path, code_dict):
    df = pd.read_csv(file_path) 

    df.rename(columns={'Description': 'Ticket_Description',
                        'Category': 'Ticket_Type'}, inplace=True)

    df['Ticket_Description'] = df['Ticket_Description'].apply(lambda x: x.lower())
    df = df.drop_duplicates().reset_index(drop=True)

    df["Clean_Ticket_Description"] = df.Ticket_Description.apply(lambda x: clean(x))
    df["Target"] = df["Ticket_Type"].map(ct.label_code)

    return df[["Clean_Ticket_Description","Target"]]