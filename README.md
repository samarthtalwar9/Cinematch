# CineMatch AI 🎬

CineMatch AI is a modern, premium, **ML-Powered Movie Recommendation System**. It features a stunning cinematic user interface built with raw HTML/CSS/JS and a lightweight, statically hosted dataset powered by machine learning algorithms (Scikit-Learn).

This project is built as a beginner-friendly, deployment-ready showcase perfect for engineering student internships. It completely eliminates complex backend hosting and database costs by shifting the machine learning output into a lightning-fast, frontend-only JSON architecture.

![CineMatch AI Screenshot](https://via.placeholder.com/1200x600/0b0f19/00f0ff?text=CineMatch+AI+Dashboard+Screenshot)

## ✨ Features

- **Premium UI/UX:** Cinematic dark mode aesthetic with glassmorphism, smooth micro-animations, and dynamic visual states.
- **Content-Based Filtering:** True AI recommendation engine that suggests movies based on shared metadata (genres, keywords, cast, and overview).
- **Offline ML Pipeline:** Heavy lifting (CountVectorizer & Cosine Similarity) is done entirely offline in Python.
- **Ultra-Fast & Lightweight:** Pure Vanilla JS frontend communicating with a static `recommendations.json` file for O(1) instantaneous lookups. No API latency.
- **Serverless Deployment:** Deployable instantly and for free on Vercel or GitHub Pages. No Python backend hosting required!

## 🛠 Tech Stack

**Frontend (Deployed):**
- HTML5
- CSS3 (Custom properties, Flexbox, CSS Grid)
- Vanilla JavaScript (ES6, Fetch API)

**Machine Learning / Data Processing (Offline):**
- Python 3.x
- Scikit-Learn (Machine Learning library)
- Pandas & NumPy (Data manipulation)

## 🧠 Machine Learning Workflow

CineMatch AI remains an authentic **ML-Powered Movie Recommendation System**. We simply shifted the compute cycle from "on-demand" to "pre-computed". Here is how it works:

1. **Data Preprocessing:** The offline Python scripts (`backend/generate_similarity.py`) load movie metadata, clean it, and combine relevant text columns (like genres, keywords, cast, and overview) into a single `tags` string for each movie.
2. **Vectorization (`CountVectorizer`):** The text tags are converted into numerical vectors. Scikit-Learn's `CountVectorizer` counts the frequency of each word, transforming the text data into a mathematical matrix.
3. **Cosine Similarity:** To determine how similar two movies are, the engine calculates the **Cosine Similarity** between their vectors.
4. **JSON Export:** A helper script (`backend/generate_recommendations_json.py`) computes the top 6 matches for every single movie in the dataset and exports the final mapping into a lightweight `frontend/recommendations.json` file.
5. **Instant Frontend Delivery:** When a user searches for a movie in the web browser, the JavaScript instantly looks up the pre-calculated ML recommendations locally—resulting in zero latency and zero server costs.

## 🚀 Setup Instructions (Local)

### 1. Clone the Repository
```bash
git clone https://github.com/samarthtalwar9/Cinematch.git
cd Cinematch
```

### 2. Run the Frontend
Because the ML logic is already pre-generated in `recommendations.json`, you do **not** need to run any Python server!
Simply open `frontend/index.html` in any modern web browser or use a tool like VS Code Live Server.

## ☁️ Deployment Instructions

The project is now a static site and can be deployed in 2 minutes:

### Vercel / GitHub Pages
1. Push your code to a GitHub repository.
2. Create a new project on [Vercel](https://vercel.com/) or enable [GitHub Pages](https://pages.github.com/).
3. Connect your repository.
4. Set the "Root Directory" to `frontend` (for Vercel).
5. Deploy! Your app will instantly serve the HTML and JSON files globally.

## 🔬 Running the Offline ML Engine (Optional)

If you wish to update the movie dataset or tweak the ML algorithm, the offline Python backend is still available in the repository:

```bash
cd backend
pip install -r requirements.txt
python generate_similarity.py          # Generates the mathematical matrix
python generate_recommendations_json.py # Exports the frontend JSON file
```

## 🔮 Future Improvements

- Automate the Python pipeline to regenerate the JSON via GitHub Actions whenever the dataset is updated.
- Expand the dataset to include accurate release years, IMDb ratings, and high-quality movie poster URLs directly in the JSON.
- Implement Collaborative Filtering (user-to-user recommendations) to supplement the current Content-Based model.
