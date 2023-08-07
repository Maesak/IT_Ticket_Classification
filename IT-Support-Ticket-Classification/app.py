import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import preprocessing as pp
import constant as ct

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



#To get pre-processed data
df = pp.get_data(ct.path,ct.label_code)

#train - Test Split
x = df['Clean_Ticket_Description']
y = df['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#=======================================================================================#
#Vectorization - 4 combinations
#were used to check the optimal performanced model - TF-IDF has been chosen and used for model development
#vec 1
c_vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
x_train_counts =c_vectorizer.fit_transform(x_train) 
x_test_counts = c_vectorizer.transform(x_test)

#vec 2
c_vectorizer_bigram = CountVectorizer(ngram_range=(1,2),stop_words=stopwords.words('english'))
x_train_counts_bigram =c_vectorizer_bigram.fit_transform(x_train) 
x_test_counts_bigram = c_vectorizer_bigram.transform(x_test)

#vec 3
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)

#vec 4
tfidf_vectorizer_bigram = TfidfVectorizer(ngram_range=(1,2),stop_words=stopwords.words('english'))
x_train_tfidf_bigram = tfidf_vectorizer_bigram.fit_transform(x_train)
x_test_tfidf_bigram = tfidf_vectorizer_bigram.transform(x_test)

#=================================================================================================#
#Model building
# models
mnb = MultinomialNB()
svc = SVC()
rfc = RandomForestClassifier()


#====================================================================================#
#Multinomial NB model
mnb.fit(x_train_tfidf, y_train)
svc.fit(x_train_tfidf, y_train)
rfc.fit(x_train_tfidf, y_train)

#prediction with Multinomial NB model
y_pred_tfidf = mnb.predict(x_test_tfidf)
tfidf_accuracy = accuracy_score(y_test, y_pred_tfidf)
print("\n Classification Accuracy with Multinomial NB model:", np.round(tfidf_accuracy,3))


#====================================================================================#
#SVM model
pred_tiidf_vectorization = svc.predict(x_test_tfidf)
acc = accuracy_score(y_test , pred_tiidf_vectorization)
print("Classification Accuracy with SVC model:",np.round(acc,3))

#====================================================================================#
#Random Forest model
pred_tfidf= rfc.predict(x_test_tfidf)
acc = accuracy_score(y_test,pred_tfidf)
print("Classification Accuracy with Random Forest model:", np.round(acc,3) , "\n")


'''
#====================================================================================#
#Predict for a new query

query = pd.Series(input("enter your issue: "))
pre_processed_query= query.apply(pp.clean)
vectorized_query =  tfidf_vectorizer.transform(pre_processed_query)
output_category = svc.predict(vectorized_query)[0]

for key, val in ct.label_code.items():
    if val==output_category:
        print(f"The issue has been predicted as the category of {key}")
'''