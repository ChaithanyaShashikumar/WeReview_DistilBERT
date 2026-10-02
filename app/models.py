from database import db


class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    review_text = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float, nullable=False)

    prediction = db.Column(db.String(50), nullable=False)
    prediction_probability = db.Column(db.Float, nullable=False)

    content_hash = db.Column(db.String(64), nullable=False, unique=True)

    blockchain_tx_hash = db.Column(db.String(100), nullable=True)

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now()
    )