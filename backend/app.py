from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
# Enable CORS so the frontend can communicate with this API
CORS(app)

print("Server starting: Loading ML models and dataset...")

try:
    # Load the datasets from the backend folder
    # Setting paths relative to the current script directory ensures it finds them
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load movie list and reset index to guarantee index matches the row number 
    # of the cosine similarity matrix
    movies = pd.read_pickle(os.path.join(base_dir, 'movie_list.pkl'))
    movies = movies.reset_index(drop=True)
    
    sim_path = os.path.join(base_dir, 'similarity.pkl')
    if not os.path.exists(sim_path):
        print("similarity.pkl not found. Generating it now...")
        import generate_similarity
        generate_similarity.main()

    # Load cosine similarity matrix
    with open(sim_path, 'rb') as f:
        similarity = pickle.load(f)
        
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Extract JSON data from the incoming POST request
        data = request.get_json()
        
        # Error handling: Check for empty request or missing movie field
        if not data or 'movie' not in data:
            return jsonify({'success': False, 'error': 'Invalid input: "movie" field is missing'}), 400
        
        movie_title = data['movie'].strip()
        print(f"Recommendation request received for: {movie_title}")
        
        # Convert search query to lowercase for case-insensitive matching
        match = movies[movies['title'].str.lower() == movie_title.lower()]
        
        # Error handling: Movie not found in dataset
        if match.empty:
            print(f"Movie not found: {movie_title}")
            return jsonify({'success': False, 'error': 'Movie not found in database'}), 404
            
        print("Movie found. Generating recommendations...")
        
        # --- Content-Based Recommendation Logic ---
        
        # 1. Find the numerical index of the requested movie
        movie_idx = match.index[0]
        
        # 2. Look up the similarity scores for this specific movie against all others
        distances = similarity[movie_idx]
        
        # 3. Sort the similarity scores in descending order
        # enumerate(distances) attaches the original index to each score: (index, score)
        # We sort by score (x[1]) and skip the first result [1:7] since it is the movie itself
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
        
        # 4. Extract movie details and format for JSON response
        recommended_movies = []
        for i in movies_list:
            movie_row = movies.iloc[i[0]]
            
            # Note: Since the dataset only contains movie_id, title, overview, and tags,
            # we inject placeholder values for year, rating, and genres to ensure
            # the frontend UI correctly renders the cinematic cards without breaking.
            recommended_movies.append({
                "title": str(movie_row['title']),
                "year": "N/A",  
                "rating": "N/A",
                "genres": ["Recommended"]
            })
            
        return jsonify({
            'success': True,
            'movies': recommended_movies
        })
        
    except Exception as e:
        # Error handling: Catch-all for backend failures
        print(f"Backend failure: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run the Flask app on all available IPs with port 5000
    app.run(host='0.0.0.0', port=5000)
