# AI Recipe Generator 🍳

An AI-powered recipe generator that suggests recipes based on ingredients you have, using machine learning and natural language processing techniques.

## Features ✨

- Generates recipes based on available ingredients
- Finds the most relevant recipes using cosine similarity
- Simple and intuitive interface
- FastAPI backend with Streamlit frontend
- Works with your custom dataset

## Technologies Used 🛠️

- Python 3.8+
- FastAPI (backend)
- Streamlit (UI)
- Scikit-learn (ML)
- Pandas (data processing)
- NLTK (text processing)

## Dataset 📊

The system uses the `test-5000.csv` dataset containing:
- `inputs`: List of ingredients
- `targets`: Complete recipes with instructions

## Installation ⚙️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/recipe-generator.git
cd recipe-generator
