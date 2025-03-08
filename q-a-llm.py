import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

# Load dataset
df = pd.read_csv('top_10000_popular_movies_tmdb.csv', encoding='ISO-8859-1')

# Fill missing values in the 'overview' column with an empty string
df['overview'] = df['overview'].fillna('')

# Load SentenceTransformer model
embedding_model = SentenceTransformer('BAAI/bge-large-en')

# Generate embeddings for movie descriptions (overview)
df['Embeddings'] = df['overview'].apply(
    lambda x: embedding_model.encode(str(x), convert_to_numpy=True) if isinstance(x, str) else np.zeros((768,)))

# Question templates
question_templates = [
    "What is the plot of '{title}'?",
    "What genre is '{title}'?",
    "When was '{title}' released?",
    "Who produced '{title}'?",
    "What was the budget of '{title}'?",
    "How much revenue did '{title}' generate?",
    "What is the runtime of '{title}'?",
    "What is the tagline of '{title}'?"
]


# Function to retrieve similar movies safely
def retrieve_similar_movies(query, k=2):
    if not isinstance(query, str) or query.strip() == "":
        return df.sample(n=k), [0.0] * k  # Return random rows if input is empty

    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    similarities = cosine_similarity([query_embedding], np.stack(df['Embeddings'].values))[0]
    top_indices = np.argsort(similarities)[-k:][::-1]  # Get top-k highest similarity scores
    return df.iloc[top_indices], similarities[top_indices]


# Generate Q-A Pairs
qa_pairs = []
for _ in range(2):  # Generate 15 pairs
    movie_row = df.sample(n=1).iloc[0]
    query_template = random.choice(question_templates)
    query = query_template.format(title=movie_row['title'])

    retrieved_movies, sim_scores = retrieve_similar_movies(movie_row['overview'])

    # Prepare LLM prompt
    prompt = f"""
    Question: {query}
    Movie Details:
    - Title: {movie_row['title']}
    - Release Date: {movie_row['release_date']}
    - Genre: {movie_row['genres']}
    - Production Companies: {movie_row['production_companies']}
    - Budget: {movie_row['budget']}
    - Revenue: {movie_row['revenue']}
    - Runtime: {movie_row['runtime']}
    - Tagline: {movie_row['tagline']}
    - Plot: {movie_row['overview']}

    Based on the above information, provide a concise answer to the question.
    """

    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])['message']['content']

    qa_pairs.append({
        "Question": query,
        "Top Match 1": retrieved_movies.iloc[0]['overview'],
        "Cosine Similarity 1": sim_scores[0],
        "Top Match 2": retrieved_movies.iloc[1]['overview'],
        "Cosine Similarity 2": sim_scores[1],
        "Generated Response": response,
        "Ground Truth": movie_row['overview']
    })

# Save to CSV
qa_results_df = pd.DataFrame(qa_pairs)
qa_results_df.to_csv('qa_pairs_results_test.csv', index=False)
print("✅ Fixed and saved Q-A pairs!")
