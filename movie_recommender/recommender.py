# recommender.py
import os
import pandas as pd 
import ast
import difflib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_TMDB = "movies.csv"  
SIM_PICKLE = "similarity.pkl"
VEC_PICKLE = "vectorizer.pkl"
MOVIES_PICKLE = "movies_df.pkl"

def load_tmdb(path):

    df = pd.read_csv(path, quotechar='"', engine='python', on_bad_lines='skip', encoding='utf-8')
    
    cols_to_keep = ['movie_id', 'title', 'cast', 'crew', 'genres', 'overview']
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()

    # helper to parse JSON-like strings
    def parse_names(x):
        if pd.isna(x) or x == '[]':
            return []
        try:
            L = ast.literal_eval(x)
            return [i['name'].replace(" ", "") for i in L]
        except Exception:
            return []

    # parse each column
    if 'cast' in df.columns:
        df['cast'] = df['cast'].apply(parse_names)
    else:
        df['cast'] = ""

    if 'crew' in df.columns:
        df['crew'] = df['crew'].apply(parse_names)
    else:
        df['crew'] = ""

    if 'genres' in df.columns:
        df['genres'] = df['genres'].apply(parse_names)
    else:
        df['genres'] = ""

    if 'overview' in df.columns:
        df['overview'] = df['overview'].fillna("").astype(str)
    else:
        df['overview'] = ""

    # build tags: cast + crew + genres + overview
    df['tags'] = (
        df['cast'].apply(lambda x: " ".join(x)) + " " +
        df['crew'].apply(lambda x: " ".join(x)) + " " +
        df['genres'].apply(lambda x: " ".join(x)) + " " +
        df['overview']
    )
    df['tags'] = df['tags'].str.lower()

    df = df[['title', 'tags']]  # keep only title & tags for recommendation
    df = df.reset_index(drop=True)
    return df

def prepare():
    # load cached objects if they exist
    if os.path.exists(MOVIES_PICKLE) and os.path.exists(SIM_PICKLE) and os.path.exists(VEC_PICKLE):
        movies = pd.read_pickle(MOVIES_PICKLE)
        with open(SIM_PICKLE, "rb") as f:
            similarity = pickle.load(f)
        with open(VEC_PICKLE, "rb") as f:
            vectorizer = pickle.load(f)
        return movies, similarity, vectorizer

    if os.path.exists(DATA_TMDB):
        movies = load_tmdb(DATA_TMDB)
    else:
        raise FileNotFoundError(f"No dataset found. Put '{DATA_TMDB}' in the folder.")

    # Vectorize tags with TF-IDF + bigrams
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(movies['tags'])
    similarity = cosine_similarity(vectors)

    # cache for faster restart
    movies.to_pickle(MOVIES_PICKLE)
    with open(SIM_PICKLE, "wb") as f:
        pickle.dump(similarity, f)
    with open(VEC_PICKLE, "wb") as f:
        pickle.dump(vectorizer, f)

    return movies, similarity, vectorizer

# prepare on import
movies, similarity, vectorizer = prepare()

def recommend(movie_name, top_n=5):
    title_list = movies['title'].str.lower().tolist()
    movie_name_low = movie_name.strip().lower()

    if movie_name_low in title_list:
        idx = movies[movies['title'].str.lower() == movie_name_low].index[0]
    else:
        close = difflib.get_close_matches(movie_name_low, title_list, n=1, cutoff=0.6)
        if close:
            idx = movies[movies['title'].str.lower() == close[0]].index[0]
        else:
            raise ValueError("Movie not found. Try an exact title or check spelling.")

    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1: top_n+1]
    recommended_titles = [movies.iloc[i[0]].title for i in distances]
    return recommended_titles

# Test the recommender
if __name__ == "__main__":
    movie = "Avatar"
    print(f"Top 5 recommendations for '{movie}':")
    print(recommend(movie))
