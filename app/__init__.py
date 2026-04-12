from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS   # 🔥 ADD THIS
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    CORS(app)   # 🔥 ADD THIS LINE

    db.init_app(app)

    # import and register routes
    from app.routes.predict_routes import predict_bp
    from app.routes.history_routes import history_bp

    app.register_blueprint(predict_bp)
    app.register_blueprint(history_bp)

    return app