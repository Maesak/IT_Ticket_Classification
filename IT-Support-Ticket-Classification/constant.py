import preprocessing as pp
import pandas as pd

from nltk.corpus import stopwords

path = r'.\data\latest_ticket_data.csv'

label_code = {
    "Application":0,
    "Database":1,
    "Network":2,
    "User Maintenance":3,
    "Security":4
}

additional_stopwords = ["i", "I","hi", "hello", "dear", "please", "thank","issue","product","purchase"]
all_stopwords = list(stopwords.words('english'))
all_stopwords.extend(additional_stopwords)
Stopwords = list(set(all_stopwords))


