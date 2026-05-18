import os
from datetime import timedelta
from tzlocal import get_localzone
from pathlib import Path


class Config:
    """Base configuration settings."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default_secret_key_for_dev')

    # 1. Path Orchestration
    # BASEDIR resolves to the project root (the folder containing the 'src' directory)
    BASEDIR = Path(__file__).resolve().parent.parent

    # 2. Directory Structure (Enforcing modular src layout)
    SRC_DIR = BASEDIR / "src"
    DATA_DIR = SRC_DIR / "data"

    # 3. Automated Provisioning
    # Ensures the platform's governed ecosystem is ready for data ingestion on boot
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    CACHE_DB_PATH = "cache_data.db"
    PIPELINE_DB_PATH= "app_data.db"
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + PIPELINE_DB_PATH
    CACHE_DATABASE_URI = "sqlite:///" + CACHE_DB_PATH

    # Database 2 (Bind): Cache Data (cache.db) - defined using custom SQLAlchemy binds
    SQLALCHEMY_BINDS = {
        'cache': CACHE_DATABASE_URI
    }

    # Suppress deprecation warning
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # LLM Settings
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
    SIMULATE_API = os.environ.get('SIMULATE_API', 'True').lower() in ('true', '1', 't')

    # RAG Settings
    VECTOR_DB_URL = os.environ.get('VECTOR_DB_URL', 'in_memory_mock')

    # --- APScheduler Configuration ---
    SCHEDULER_API_ENABLED = True  # Allows you to see job status via /scheduler/jobs
    SCHEDULER_TIMEZONE = str(get_localzone())

class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    SESSION_COOKIE_SECURE = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    SESSION_COOKIE_SECURE = True  # Require HTTPS for cookies
    # In production, secrets must be set via environment variables
