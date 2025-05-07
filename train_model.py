import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def prepare_data():
    df = pd.read_csv('C:/Users/LENOVO/Desktop/ai_recipe_generator/data/test-5000.csv', sep='\t')

    df['inputs'] = df['inputs'].str.lower().str.replace(',', '')
    df['targets'] = df['targets'].str.lower()
    
    return df

def train_model():
    df = prepare_data()

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['inputs'])

    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open('models/recipe_model.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    return df, vectorizer, tfidf_matrix

if __name__ == '__main__':
    df, vectorizer, tfidf_matrix = train_model()
    print("Model trained and saved successfully!")