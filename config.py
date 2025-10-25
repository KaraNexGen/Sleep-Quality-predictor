"""
Configuration settings for Sleep Quality Predictor
"""

import os
from pathlib import Path

# Base configuration
BASE_DIR = Path(__file__).parent
DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'

# Flask configuration
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = DEBUG
    TESTING = False
    
    # Model configuration
    MODEL_PATH = BASE_DIR / 'models'
    MODEL_FILENAME = 'sleep_models.joblib'
    
    # Data configuration
    DATASET_SIZE = int(os.getenv('DATASET_SIZE', '1000'))
    SAMPLE_DATA_FILE = BASE_DIR / 'sample_sleep_data.csv'
    
    # Feature configuration
    FEATURE_NAMES = [
        'sleep_duration', 'bedtime_hour', 'wakeup_hour', 'caffeine_intake',
        'exercise_duration', 'screen_time_before_bed', 'stress_level',
        'mood_before_sleep', 'sleep_interruptions', 'room_temperature',
        'noise_level', 'light_exposure', 'alcohol_intake', 'meal_timing'
    ]
    
    # Sleep quality thresholds
    SLEEP_QUALITY_THRESHOLDS = {
        'Good': 25,
        'Average': 18,
        'Poor': 0
    }
    
    # Recommendation priorities
    RECOMMENDATION_PRIORITIES = {
        'sleep_duration': 10,
        'stress_level': 9,
        'screen_time_before_bed': 8,
        'bedtime_hour': 7,
        'exercise_duration': 6,
        'caffeine_intake': 5,
        'room_temperature': 4,
        'noise_level': 3,
        'light_exposure': 2,
        'alcohol_intake': 1
    }

# Development configuration
class DevelopmentConfig(Config):
    DEBUG = True
    DATASET_SIZE = 500

# Production configuration
class ProductionConfig(Config):
    DEBUG = False
    DATASET_SIZE = 2000

# Testing configuration
class TestingConfig(Config):
    TESTING = True
    DATASET_SIZE = 100

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
