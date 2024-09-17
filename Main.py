import streamlit as st
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Download necessary NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('brown')

# Load a subset of the Brown Corpus sentences
sentences = nltk.corpus.brown.sents(categories='news')[:1000]  # Using only the first 1000 sentences for demonstration

# Function to convert sentences into continuous text and label sentence boundaries
def create_sentence_boundary_data(sentences):
    words = []
    labels = []
    
    for sentence in sentences:
        for i, word in enumerate(sentence):
            words.append(word)
            if i == len(sentence) - 1:  # Last word in the sentence
                labels.append(1)  # 1 indicates the end of a sentence (EOS)
            else:
                labels.append(0)  # 0 indicates continuation of a sentence
    
    return words, labels

# Function to extract features, including POS tagging and word length
def extract_features(words):
    features = []
    pos_tags = nltk.pos_tag(words)  # Get part-of-speech tags
    
    for i, (word, pos) in enumerate(pos_tags):
        word_features = {}
        
        # Feature 1: The word itself (we'll vectorize this later)
        word_features['word'] = word
        
        # Feature 2: Check if it's a punctuation mark
        word_features['is_punctuation'] = int(word in ['.', '?', '!'])  # Convert to int
        
        # Feature 3: Check if the next word is capitalized
        word_features['next_word_capitalized'] = int((i < len(words) - 1) and words[i + 1][0].isupper())  # Convert to int
        
        # Feature 4: Word length
        word_features['word_length'] = len(word)
        
        # Feature 5: Part-of-speech (POS) tag
        word_features['pos_tag'] = pos
        
        # Feature 6: Is the word the first word in the sentence?
        word_features['is_first_word'] = int(i == 0)  # Convert to int
        
        # Feature 7: Is the word the last word in the sentence?
        word_features['is_last_word'] = int(i == len(words) - 1)  # Convert to int
        
        features.append(word_features)
    
    return features

# Prepare the data for the model
words, labels = create_sentence_boundary_data(sentences)
features = extract_features(words)
df = pd.DataFrame(features)

# Vectorize the 'word' and 'pos_tag' features using CountVectorizer
vectorizer_word = CountVectorizer()
vectorizer_pos = CountVectorizer()

X_word = vectorizer_word.fit_transform(df['word'])
X_pos = vectorizer_pos.fit_transform(df['pos_tag'])

# Convert vectorized features to dense arrays
X_word_dense = X_word.toarray()
X_pos_dense = X_pos.toarray()

# Combine the vectorized 'word' and 'pos_tag' features with other features
X = np.hstack([
    X_word_dense, 
    X_pos_dense, 
    df[['is_punctuation', 'next_word_capitalized', 'word_length', 'is_first_word', 'is_last_word']].values
])

# Target labels (EOS labels)
y = np.array(labels)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
model.fit(X, y)

# Streamlit App
st.title('Sentence Boundary Detection')

st.write('Enter a sentence below and see if the model detects the sentence boundaries correctly.')

input_text = st.text_area("Input Text")

if st.button('Predict'):
    if input_text:
        # Tokenize and extract features for the input text
        words_input = nltk.word_tokenize(input_text)
        features_input = extract_features(words_input)
        df_input = pd.DataFrame(features_input)
        
        X_word_input = vectorizer_word.transform(df_input['word'])
        X_pos_input = vectorizer_pos.transform(df_input['pos_tag'])
        
        X_word_input_dense = X_word_input.toarray()
        X_pos_input_dense = X_pos_input.toarray()
        
        X_input = np.hstack([
            X_word_input_dense,
            X_pos_input_dense,
            df_input[['is_punctuation', 'next_word_capitalized', 'word_length', 'is_first_word', 'is_last_word']].values
        ])
        
        # Make predictions
        y_pred_input = model.predict(X_input)
        
        # Display predictions
        st.write('Predictions:')
        for i, word in enumerate(words_input):
            boundary = "End of Sentence" if y_pred_input[i] == 1 else "Continuation"
            st.write(f"'{word}': {boundary}")
    else:
        st.write('Please enter some text to predict.')
