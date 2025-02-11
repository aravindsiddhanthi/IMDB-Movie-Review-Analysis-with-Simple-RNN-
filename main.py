## Step 1 : Import Libraries and Load the Model

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word Index
word_Index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_Index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Function
#Function to decode reviews
def code_review(encoded_review):
    return ' '.join([reverse_word_index.get(i -3, '?') for i in encoded_review])


# function to preporcess user input (basically it will give index with padding sequence)
def preprocess_text(text):
    words = text.lower().split()
    encoded_review =  [word_Index.get(word,2) +3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review



## streamlit app

import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User Input 
user_input = st.text_area('Movie Review')

## now to create a button

if st.button('Classify'):

    preprocess_input=preprocess_text(user_input)

    ## Make prediction
    prediction=model.predict(preprocess_input)
    sentiment  = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    ## Dosply the result 
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')


