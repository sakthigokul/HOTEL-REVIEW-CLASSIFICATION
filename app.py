# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:19:47 2021

@author: smile
"""

import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('classifiers.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(textdata):   
 
    # Making predictions 
    prediction = classifier.predict( 
        [[textdata]])
     
   
    return prediction

def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
  
    EMAIL_TEMPLATE = st.text_input("which template you want?") 
   
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(textdata) 
        st.success('Your email template is {}'.format(result))
        print(textdata)
     
