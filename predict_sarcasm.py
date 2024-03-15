# Script 2: Load Model and Make Predictions

# import numpy as np
import pandas as pd
# import os
import re
import string
from keras.models import model_from_json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.models import load_model
import tensorflow as tf
import logging

# Set TensorFlow logging level to only display errors
tf.get_logger().setLevel(logging.ERROR)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer_obj = pickle.load(tokenizer_file)

max_length = 25

# Load the model architecture
json_file = open('sarcasm_detection_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("sarcasm_detection_model_weights.h5")
# print("Model loaded successfully.")

def clean_text(text):
    # Assuming you have a function called clean_text
    # Implement the cleaning logic as needed
    # For example: return text.lower()
    text = text.lower()
    
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text

# def predict_sarcasm(s):
#     x_final = pd.DataFrame({"headline": [s]})
#     test_lines = clean_text(x_final)
#     test_lines = tokenizer_obj.texts_to_sequences(test_lines)
#     test_review_pad = pad_sequences(test_lines, maxlen=max_length, padding='post')
#     pred = loaded_model.predict(test_review_pad)
#     pred *= 100
#     if pred[0][0] >= 50:
#         return "Sentence contains sarcasm!"
#     else:
#         return "Sentence does not contain sarcasm."
def CleanTokenize(df):
    head_lines = list()
    lines = df["headline"].values.tolist()

    for line in lines:
        line = clean_text(line)
        tokens = word_tokenize(line)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        words = [w for w in words if not w in stop_words]
        head_lines.append(words)
    return head_lines

def predict_sarcasm(s):
    x_final = pd.DataFrame({"headline": [s]})
    test_lines = x_final['headline'].apply(clean_text)
    test_lines = CleanTokenize(pd.DataFrame({"headline": test_lines}))
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    pred = loaded_model.predict(test_review_pad)
    pred *= 100
    if pred[0][0] >= 50:
        return "Sentence contains sarcasm!"
    else:
        return "Sentence does not contain sarcasm."


# Example usage
result = predict_sarcasm("You just broke my car window. Great job.")
print(result)