Movie Recommendation System (Content-Based)
Overview

This project is a content-based movie recommendation system that suggests movies similar to a given input movie.
It uses metadata such as cast, crew, genres, and plot overview to compute similarity between movies.
The system is exposed as a Flask API so recommendations can be retrieved in real time.

Features

--Content-based filtering using movie metadata (cast, crew, genres, overview).

--TF-IDF Vectorization (with unigrams & bigrams) for richer feature representation.

--Cosine Similarity to measure similarity between movies.

--Fuzzy matching for handling misspelled movie titles.

--Exposed as a REST API (Flask) for real-time recommendations.

 Tech Stack

--Python (Pandas, NumPy, scikit-learn)

--Flask (for API)

--TF-IDF Vectorizer + Cosine Similarity

--Dataset: TMDB Movies Dataset

Project Structure
movie-recommender/
│── app.py              # Flask API
│── recommender.py      # Recommendation logic (TF-IDF + Cosine Similarity)
│── requirements.txt    # Dependencies
│── movies.csv          # Dataset (TMDB or MovieLens)
│── README.md           # Project documentation

Run the Flask app:

--python app.py

 Usage:

Open browser or Postman and go to:

http://127.0.0.1:5000/recommend?movie=Avatar

Example output:

{
  "query": "Avatar",
  "recommendations": [
    "Guardians of the Galaxy",
    "Star Trek",
    "The Fifth Element",
    "Interstellar",
    "John Carter"
  ]
}


🔹 Future Improvements

--Use word embeddings (Word2Vec / BERT) for semantic similarity.

--Extend to a Hybrid Recommender (content + collaborative filtering).

--Deploy using Docker + AWS/GCP for cloud hosting.

--Add a simple Streamlit web UI for user-friendly interaction.

Deploy using Docker + AWS/GCP for cloud hosting.

Add a simple Streamlit web UI for user-friendly interaction.
