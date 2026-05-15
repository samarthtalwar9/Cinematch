# CineMatch AI 🎬

CineMatch AI is a modern, premium, AI-powered movie recommendation web application. It features a stunning cinematic user interface built with raw HTML/CSS/JS and a lightweight Machine Learning backend powered by Python, Flask, and Scikit-Learn.

This project is built as a beginner-friendly, deployment-ready showcase perfect for engineering student internships. It avoids unnecessary complexity (like databases or user authentication) and focuses squarely on beautiful design, core machine learning principles, and clean API integration.

![CineMatch AI Screenshot](https://via.placeholder.com/1200x600/0b0f19/00f0ff?text=CineMatch+AI+Dashboard+Screenshot)

## ✨ Features

- **Premium UI/UX:** Cinematic dark mode aesthetic with glassmorphism, smooth micro-animations, and dynamic visual states.
- **Content-Based Filtering:** AI recommendation engine that suggests movies based on shared metadata (genres, keywords, cast, and overview).
- **Fast & Lightweight:** No complex frameworks—just pure Vanilla JS frontend communicating seamlessly with a Python API.
- **Deployment Ready:** Configured with a `Procfile` and `requirements.txt` for easy hosting on platforms like Render and Vercel.

## 🛠 Tech Stack

**Frontend:**
- HTML5
- CSS3 (Custom properties, Flexbox, CSS Grid)
- Vanilla JavaScript (ES6, Fetch API)

**Backend:**
- Python 3.x
- Flask (Web framework)
- Flask-CORS (Cross-Origin Resource Sharing)
- Scikit-Learn (Machine Learning library)
- Pandas & NumPy (Data manipulation)

## 🧠 Machine Learning Workflow

The recommendation engine uses a **Content-Based Filtering** approach. Here is how it works:

1. **Data Preprocessing:** The backend script (`generate_similarity.py`) loads movie metadata (`movie_list.pkl`), cleans it, and combines relevant text columns (like genres, keywords, cast, and overview) into a single `tags` string for each movie.
2. **Vectorization (`CountVectorizer`):** The text tags are converted into numerical vectors. Scikit-Learn's `CountVectorizer` counts the frequency of each word, transforming the text data into a mathematical matrix that the computer can understand.
3. **Cosine Similarity:** To determine how similar two movies are, the engine calculates the **Cosine Similarity** between their vectors. This measures the angle between the vectors—the smaller the angle, the higher the similarity score.
4. **Prediction:** When a user searches for a movie, the Flask API finds its index, looks up its similarity scores against all other movies, and returns the top 6 matches in descending order.

## 🚀 Setup Instructions (Local)

### 1. Clone the Repository
```bash
git clone https://github.com/samarthtalwar9/Cinematch.git
cd Cinematch
```

### 2. Run the Backend
Ensure you have Python installed.
```bash
cd backend
pip install -r requirements.txt
python app.py
```
The Flask server will start running on `http://localhost:5000`.

### 3. Run the Frontend
Simply open `frontend/index.html` in any modern web browser. No `npm install` or local server is required for the frontend!

## ☁️ Deployment Instructions

### Backend (Render)
1. Push your code to a GitHub repository.
2. Create a new "Web Service" on [Render](https://render.com/).
3. Connect your repository.
4. Set the Build Command to: `pip install -r backend/requirements.txt && python backend/generate_similarity.py`
   *(Note: `similarity.pkl` is generated dynamically during build because it exceeds GitHub's 100MB file limit).*
5. Set the Start Command to: `gunicorn backend.app:app`
6. Deploy! Render will provide you with a live URL (e.g., `https://cinematch-backend.onrender.com`).

### Frontend (Vercel)
1. Open `frontend/index.html` in your code editor.
2. Update the `API_URL` variable at the top of the JavaScript block:
   ```javascript
   const API_URL = "https://your-render-backend-url.com/recommend";
   ```
3. Push the changes to GitHub.
4. Create a new project on [Vercel](https://vercel.com/) and connect your repository.
5. Set the "Root Directory" to `frontend`.
6. Deploy! 

## 🔮 Future Improvements

- Replace the `.pkl` file with a live cloud database (e.g., Firebase or PostgreSQL) for real-time dataset updates.
- Expand the dataset to include accurate release years, IMDb ratings, and high-quality movie poster URLs.
- Implement Collaborative Filtering (user-to-user recommendations) to supplement the current Content-Based model.
