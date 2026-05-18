# src/__init__.py

import os
from flask import Flask
from flask_wtf import CSRFProtect
from flask_sqlalchemy import SQLAlchemy
from src.config import DevelopmentConfig, ProductionConfig


db = SQLAlchemy()
csrf = CSRFProtect()

# --- Environment setup ---
# Determine which configuration to use
if os.environ.get('FLASK_ENV') == 'production':
    config = ProductionConfig
else:
    config = DevelopmentConfig

def create_app(config_class=config):
    """
    The Flask application factory function.
    Initializes and configures the Flask application instance.
    """
    # 1. Initialize core application
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 2. Initialize extensions
    db.init_app(app)
    csrf.init_app(app)

    with app.app_context():
        from src.routes import dashboard_bp
        app.register_blueprint(dashboard_bp)

        # IMPORTANT: Import models here to ensure tables are created
        from src import models
        return app
