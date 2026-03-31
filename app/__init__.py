from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    # import and register routes
    from app.routes.predict_routes import predict_bp
    from app.routes.history_routes import history_bp

    app.register_blueprint(predict_bp)
    app.register_blueprint(history_bp)

    return app