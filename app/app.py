import hashlib

from flask import Flask, request, jsonify
from database import db
from models import Review
from feature_engineering import build_behavioral_features
from predictor import build_hybrid_features, predict


app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///wereview.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)


with app.app_context():
    db.create_all()


@app.route("/")
def home():
    return "WeReview Backend is Running!"


@app.route("/predict", methods=["POST"])
def predict_review():
    data = request.get_json()

    review_text = data.get("review_text", "")
    rating = data.get("rating", 0)

    behavioral_features = build_behavioral_features(
        review_text=review_text,
        rating=rating,
    )

    hybrid_features = build_hybrid_features(
        review_text,
        behavioral_features,
    )

    prediction, probability = predict(hybrid_features)

    # Generate SHA-256 hash
    content_hash = hashlib.sha256(
        review_text.encode("utf-8")
    ).hexdigest()

    # Save review and prediction to database
    review = Review(
        review_text=review_text,
        rating=rating,
        prediction=prediction,
        prediction_probability=float(probability),
        content_hash=content_hash
    )

    db.session.add(review)
    db.session.commit()

    return jsonify({
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "database_id": review.id,
        "content_hash": content_hash
    })


if __name__ == "__main__":
    app.run(debug=True)