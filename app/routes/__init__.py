from flask import Flask

def create_app():
    app = Flask(__name__)

    from app.routes.main_routes import main
    from app.routes.predict_routes import predict_bp

    app.register_blueprint(main)
    app.register_blueprint(predict_bp)

    return app