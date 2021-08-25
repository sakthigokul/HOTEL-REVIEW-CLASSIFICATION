# -*- coding: utf-8 -*-
import pickle
#from flasgger import Swagger
import streamlit as st 
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from cleantext import clean
#app=Flask(__name__)
#Swagger(app)


#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def classify_utterance(text):
    tfidf = pickle.load(open('tfidf.pickle', 'rb'))

    # load the model
    model = pickle.load(open('linear_classifier.pickle', 'rb'))
    
    sentiment_map = {'Negative':0, 'Positive':1, 'Compliant':2}
    def get_name(sentiment_id):
  
            for sentiment, id_ in sentiment_map.items():
                if id_ == sentiment_id:
                    return sentiment
    def get_sentiment(text):
        
        
        sentiment_id = model.predict(tfidf.transform([text]))

        return get_name(sentiment_id)
        
        
    
        
    def get_noun(text):
    
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)    
        pos_tags = nltk.pos_tag(tokens)
        nouns = []
        for word, tag in pos_tags:
            if tag == "NN" or tag == "NNP" or tag == "NNS":
               nouns.append(word)
        return nouns




    def top_pos_word(text):
  
        pos_polarity = dict()
        for word in nltk.word_tokenize(text):
            pos_score = SentimentIntensityAnalyzer().polarity_scores(word)['pos']
            if word not in pos_polarity:
                pos_polarity[word] = pos_score
            else:
                pos_polarity[word] += pos_score
        top_word = max(pos_polarity, key=pos_polarity.get)
        return top_word

    


    def top_neg_word(text):
  
        neg_polarity = dict()
        for word in nltk.word_tokenize(text):
            neg_score = SentimentIntensityAnalyzer().polarity_scores(word)['neg']
            if word not in neg_polarity:
                neg_polarity[word] = neg_score
            else:
                neg_polarity[word] += neg_score
        top_word = max(neg_polarity, key=neg_polarity.get)
        return top_word


    def sentiment_analysis(text):
   
        text = clean(text)
        sentiment = get_sentiment(text)
    
        if sentiment == 'Positive':
            nouns = get_noun(text)
            return(f'Sentiment: {sentiment}'),(f'Positive word: {top_pos_word(text)}'),(f'Cause of positivity: {nouns}')
            return(f'Cause of positivity: {nouns}')
        elif sentiment == 'Negative':
            nouns = get_noun(text)
            return(f'Sentiment: {sentiment}'),(f'Negative word: {top_neg_word(text)}'),(f'Cause of negativity: {nouns}')
            return(f'Cause of negativity: {nouns}')
        return(f'Sentiment: {sentiment}')
    
    # make a prediction
    return(sentiment_analysis(text))



def main():
    st.title("SENTIMENTAL ANALYSIS")

    page_bg_img = '''
             <style>
                body {
                    
                       background-image: url("https://www.stkconf.org/wp-content/uploads/2018/10/Web-Page-Background-Color.jpg");
                  background-size: cover;
      
                       }      
                </style>
            '''
    
    st.markdown(page_bg_img,unsafe_allow_html=True)
    st.header("REVIEW CLASSIFICATION ML App")
    text = st.text_input( "USER TEXT","")
  
    result=""
    if st.button("REVIEW SENTIMENT ANALYSIS"):
        result=classify_utterance(text)
    
        st.success('{}'.format(result))
   
if __name__=='__main__':
    main()
    
    
