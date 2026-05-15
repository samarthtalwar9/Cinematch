import pandas as pd
import pickle
import json
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load dataset
    movie_list_path = os.path.join(base_dir, 'movie_list.pkl')
    sim_path = os.path.join(base_dir, 'similarity.pkl')
    
    print(f"Loading {movie_list_path}...")
    movies = pd.read_pickle(movie_list_path)
    movies = movies.reset_index(drop=True)
    movies = movies.head(2000) # Ensure it matches similarity matrix dimensions
    
    print(f"Loading {sim_path}...")
    with open(sim_path, 'rb') as f:
        similarity = pickle.load(f)
        
    print("Generating recommendations JSON mapping...")
    recommendations_dict = {}
    
    # Pre-compute top 6 recommendations for each movie
    for movie_idx in range(len(movies)):
        movie_title = str(movies.iloc[movie_idx]['title'])
        # Use lowercase title as key for case-insensitive lookup later
        key = movie_title.lower().strip()
        
        distances = similarity[movie_idx]
        # Skip the movie itself [1:7]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
        
        recommended_movies = []
        for i in movies_list:
            movie_row = movies.iloc[i[0]]
            recommended_movies.append({
                "title": str(movie_row['title']),
                "year": "N/A",  
                "rating": "N/A",
                "genres": ["Recommended"]
            })
            
        recommendations_dict[key] = recommended_movies
        
    # Output to frontend directory
    output_path = os.path.join(base_dir, '..', 'frontend', 'recommendations.json')
    print(f"Saving {len(recommendations_dict)} movies to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(recommendations_dict, f, ensure_ascii=False)
        
    print("Generation complete!")

if __name__ == '__main__':
    main()
