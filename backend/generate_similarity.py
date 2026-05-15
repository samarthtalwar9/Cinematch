import pandas as pd
import pickle
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load movie data
    print("Loading dataset from movie_list.pkl...")
    try:
        movies = pd.read_pickle(os.path.join(base_dir, 'movie_list.pkl'))
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Reset index to ensure positional mapping matches the similarity matrix rows
    movies = movies.reset_index(drop=True)

    # --- Deployment Optimization for Free-Tier Hosting ---
    # Memory Efficiency: Generating a 5000x5000 matrix takes ~200MB of RAM, which often crashes 
    # free-tier cloud platforms like Render. We limit the dataset to the top 2000 movies to 
    # generate a significantly smaller similarity matrix (~30MB) that comfortably fits in memory.
    print("Optimizing dataset for cloud deployment...")
    movies = movies.head(2000)

    # 2. Analyze dataset structure and identify available columns
    available_columns = movies.columns.tolist()
    print(f"Columns detected in dataset: {available_columns}")

    # 3 & 4. Clean and preprocess text data
    # Create a combined text feature called `tags` using available metadata columns
    metadata_cols = ['genres', 'keywords', 'cast', 'overview']
    cols_to_use = [col for col in metadata_cols if col in available_columns]
    
    # Handle column differences gracefully
    if len(cols_to_use) > 0:
        movies['tags'] = ""
        for col in cols_to_use:
            # lowercase conversion and remove null values
            movies[col] = movies[col].fillna("").astype(str).str.lower()
            # combine metadata into a single tags column
            movies['tags'] = movies['tags'] + " " + movies[col]
        
        # Clean up whitespace
        movies['tags'] = movies['tags'].str.strip()
    elif 'tags' in available_columns:
        print("Specific metadata columns not found, but a 'tags' column already exists. Using it.")
        movies['tags'] = movies['tags'].fillna("").astype(str).str.lower()
    else:
        print("Error: No suitable columns found to create 'tags'.")
        return

    # 5. Convert text into vectors using CountVectorizer
    # Vectorization Explanation:
    # Computers cannot understand raw text, so we convert text into numerical vectors.
    # CountVectorizer counts the frequency of words in the 'tags' column.
    # We use stop_words='english' to remove common words (like 'the', 'and', 'is') 
    # that don't contribute to the meaning of the movie.
    print("Starting vectorization using CountVectorizer...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    print("Vectorization completed.")

    # 6. Calculate similarity using cosine_similarity
    # Cosine Similarity Explanation:
    # Cosine similarity calculates the cosine of the angle between two vectors.
    # A smaller angle (cosine similarity closer to 1) means two movies share 
    # many of the same words in their tags, indicating they are very similar.
    print("Calculating cosine similarity matrix...")
    similarity = cosine_similarity(vectors)
    print("Similarity matrix generated.")

    # 7. Save the similarity matrix as similarity.pkl
    print("Saving similarity matrix to similarity.pkl...")
    with open(os.path.join(base_dir, 'similarity.pkl'), 'wb') as f:
        pickle.dump(similarity, f)
    print("Pickle file saved.")

    # 11. End print
    print("\nsimilarity.pkl generated successfully\n")

    # 13. Small test block
    # Recommendation Logic Explanation:
    # To recommend movies, we first find the index of the requested movie.
    # We then look up that row in the similarity matrix to get similarity scores 
    # against all other movies. Finally, we sort those scores in descending order 
    # and return the top matches (excluding the first match, which is the movie itself).
    target_movie = "Interstellar"
    if 'title' in available_columns:
        if target_movie in movies['title'].values:
            print(f"--- Test Block: Top 5 recommendations for '{target_movie}' ---")
            
            # Load the similarity.pkl file
            with open(os.path.join(base_dir, 'similarity.pkl'), 'rb') as f:
                loaded_similarity = pickle.load(f)
            
            # Find the positional index of the target movie
            movie_idx = movies[movies['title'] == target_movie].index[0]
            
            # Get the similarity scores for the target movie
            distances = loaded_similarity[movie_idx]
            
            # Enumerate to keep indices, sort by similarity score (descending), take top 5
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
            
            # Print the top 5 similar movies
            for i in movies_list:
                print(movies.iloc[i[0]]['title'])
            print("---------------------------------------------------------")
        else:
            print(f"Test Block: '{target_movie}' not found in the dataset.")
    else:
        print("Test Block: 'title' column not found, cannot test movie recommendation.")

if __name__ == "__main__":
    main()
