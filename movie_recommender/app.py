# app.py
from flask import Flask, request, jsonify
from recommender import recommend

app = Flask(__name__)

@app.route("/")
def home():
    return "<h3>Movie Recommender API</h3><p>Use /recommend?movie=Movie+Name</p>"

@app.route("/recommend", methods=["GET"])
def recommend_route():
    movie = request.args.get("movie")
    if not movie:
        return jsonify({"error": "Please provide ?movie=Movie+Name"}), 400
    try:
        recs = recommend(movie, top_n=5)
        return jsonify({"query": movie, "recommendations": recs}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
