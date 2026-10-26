
import hashlib

from flask import Flask, request, jsonify, render_template
from sqlalchemy.exc import IntegrityError

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
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_review():
    data = request.get_json(silent=True) or {}

    review_text = data.get("review_text", "")
    rating = data.get("rating", 0)

    if not isinstance(review_text, str) or len(review_text.strip()) < 10:
        return jsonify({
            "error": "Please enter a review with at least 10 characters."
        }), 400

    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return jsonify({"error": "Rating must be a number from 1 to 5."}), 400

    if not 1 <= rating <= 5:
        return jsonify({"error": "Rating must be between 1 and 5."}), 400

    review_text = review_text.strip()

    # Generate behavioral features
    behavioral_features = build_behavioral_features(
        review_text=review_text,
        rating=rating,
    )

    # Generate DistilBERT embeddings + behavioral features
    hybrid_features = build_hybrid_features(
        review_text,
        behavioral_features,
    )

    # Predict using the trained XGBoost classifier
    prediction, probability = predict(hybrid_features)

    # Generate SHA-256 hash of the review text
    content_hash = hashlib.sha256(
        review_text.encode("utf-8")
    ).hexdigest()

    # Save the review and prediction
    review = Review(
        review_text=review_text,
        rating=rating,
        prediction=prediction,
        prediction_probability=float(probability),
        content_hash=content_hash,
    )

    try:
        db.session.add(review)
        db.session.commit()

    except IntegrityError:
        db.session.rollback()

        # The same review text already exists
        existing_review = Review.query.filter_by(
            content_hash=content_hash
        ).first()

        if existing_review:
            return jsonify({
                "error": "This exact review has already been submitted.",
                "prediction": existing_review.prediction,
                "probability": round(
                    float(existing_review.prediction_probability), 4
                ),
                "database_id": existing_review.id,
                "content_hash": existing_review.content_hash,
                "duplicate": True,
            }), 409

        return jsonify({
            "error": "The review could not be saved. Please try again."
        }), 500

    return jsonify({
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "database_id": review.id,
        "content_hash": content_hash,
        "duplicate": False,
    })


if __name__ == "__main__":
    app.run(debug=True)
