"""
Simple Content-Based Recommendation

Dataset source: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies/data
Author: Janice Fong
LinkedIn: https://www.linkedin.com/in/janice-ziqing-fong/

"""

# Constant value
fileName = "TMDB_movie_dataset_shorten.csv"
top_n_recommend = 5

# Code to install python libraries : pip install contractions streamlit pandas contractions nltk scikit-learn
#import Libraries
import pandas as pd
import streamlit as st

#Python libraries for data processing
import contractions
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import data & Data Cleaning
def loadData():
    df = pd.read_csv(fileName)
    print('df check null', df.isnull().sum())
    df.fillna('',inplace=True)
    print('df shape', df.shape)
    print('df dtypes', df.dtypes)
    df['combined_features'] = df['genres'] + " " + df['overview'] + " " + df['keywords']
    return df

# Data Processing
# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing texts
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Removing HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Removing URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Removing special characters and punctuation
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Removing numbers
    text = re.sub(r'\d+', '', text)

    # Performing contractions
    text = contractions.fix(text)

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stop words
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming and Lemmatization
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Removing non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]

    text = ' '.join(tokens)
    return text


# Generate Recommendation
def generate_recommendation(text):
    data = loadData()
    
    # Apply preprocessing  
    processed_data = data['combined_features'].apply(preprocess_text)
    process_text = preprocess_text(text)
    
    #TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(processed_data.tolist())
    result = vectorizer.transform([process_text])

    # find similarity
    similarity_scores = cosine_similarity(result, tfidf_matrix).flatten()

    #sort most similar
    top_indices = similarity_scores.argsort()[::-1][:top_n_recommend]
    recommendations = data.iloc[top_indices]
    print(recommendations)

    return recommendations

# Test recommendations
# testData = 'I want to watch action movies that a spy have the mission to rescue kidnapping people'
#
# recommendations = generate_recommendation(testData)
# for idx, row in recommendations.iterrows():
#     print('==============')
#     print(f"Recommended: {row['title']} - {row['overview']} - {row['genres']} - { row['keywords']}")


# UI with Streamlit
# Define the main function that sets up the layout and functionality of the Streamlit app.
def main():       
    # Create input fields where users can enter data:
    st.title('Movie Recommended System')
    text = st.text_area('Enter your preference: ','')
      
    # When the 'Recommend' button is clicked, call function and display the result.
    if text!= '' and st.button("Recommend"): 
        if text:
            result = generate_recommendation(text)

            # Show the result.
            st.success('Your recommendation') 
            num = 1
            for idx,row in result.iterrows():
                html_temp = """ 
                    <div style ="padding:13px"> 
                        <h3>Movie """+ str(num)+"""</h3>
                        <p>
                            Title: """+ row['title']+ """ <br/>
                            Genres: """+ row['genres']+""" <br/>
                            Languages: """+ row['spoken_languages']+""" <br/>
                            Overview: <br/>"""+ row['overview']+ """
                        </p>
                    </div> 
                """
                st.markdown(html_temp, unsafe_allow_html = True)
                num+=1
        else:
            st.warning("Please enter a note to summarize.")

# Check if the script is being run directly and call the main function to run the app.
if __name__=='__main__': 
    main()
# Run the Streamlit app by executing the app.py script using Streamlit's CLI.
# Use this code to run the system > streamlit run app.py 
