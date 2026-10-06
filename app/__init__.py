from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

db = SQLAlchemy()


def create_app():
    app = Flask(
        __name__,
        template_folder="../frontend/templates"
    )

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    CORS(app)

    db.init_app(app)

    # Import and register routes
    from app.routes.predict_routes import predict_bp
    from app.routes.history_routes import history_bp

    app.register_blueprint(predict_bp)
    app.register_blueprint(history_bp)

    # Serve the PhishShield web interface
    @app.route("/")
    def home():
        return render_template("index.html")

    return app