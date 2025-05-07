from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/recipe_model.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

df = pd.read_csv('data/test-5000.csv', sep='\t')
df['inputs'] = df['inputs'].str.lower().str.replace(',', '')

class RecipeRequest(BaseModel):
    ingredients: str

@app.post("/generate_recipe")
async def generate_recipe(request: RecipeRequest):
    ingredients = request.ingredients.lower()

    input_vec = vectorizer.transform([ingredients])

    cosine_similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
    
    # Get top 3 most similar recipes
    top_indices = np.argsort(cosine_similarities)[-3:][::-1]
    results = []
    
    for idx in top_indices:
        results.append({
            'ingredients': df.iloc[idx]['inputs'],
            'recipe': df.iloc[idx]['targets'],
            'similarity_score': float(cosine_similarities[idx])
        })
    
    return {'recipes': results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)