from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class SleepQualityPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = [
            'sleep_duration', 'bedtime_hour', 'wakeup_hour', 'caffeine_intake',
            'exercise_duration', 'screen_time_before_bed', 'stress_level',
            'mood_before_sleep', 'sleep_interruptions', 'room_temperature',
            'noise_level', 'light_exposure', 'alcohol_intake', 'meal_timing'
        ]
        self.load_or_create_models()
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic sleep quality dataset"""
        np.random.seed(42)
        
        data = {
            'sleep_duration': np.random.normal(7.5, 1.5, n_samples).clip(3, 12),
            'bedtime_hour': np.random.normal(22.5, 2, n_samples).clip(18, 26),
            'wakeup_hour': np.random.normal(6.5, 1.5, n_samples).clip(4, 10),
            'caffeine_intake': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2]),
            'exercise_duration': np.random.exponential(30, n_samples).clip(0, 180),
            'screen_time_before_bed': np.random.exponential(60, n_samples).clip(0, 300),
            'stress_level': np.random.uniform(0, 10, n_samples),
            'mood_before_sleep': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
            'sleep_interruptions': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'room_temperature': np.random.normal(20, 3, n_samples).clip(15, 30),
            'noise_level': np.random.uniform(0, 10, n_samples),
            'light_exposure': np.random.uniform(0, 10, n_samples),
            'alcohol_intake': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'meal_timing': np.random.normal(19, 2, n_samples).clip(16, 23)
        }
        
        df = pd.DataFrame(data)
        
        # Create sleep quality labels based on rules
        sleep_quality = []
        for _, row in df.iterrows():
            score = 0
            
            # Sleep duration scoring
            if 7 <= row['sleep_duration'] <= 9:
                score += 3
            elif 6 <= row['sleep_duration'] < 7 or 9 < row['sleep_duration'] <= 10:
                score += 2
            else:
                score += 1
            
            # Bedtime scoring
            if 22 <= row['bedtime_hour'] <= 23:
                score += 3
            elif 21 <= row['bedtime_hour'] < 22 or 23 < row['bedtime_hour'] <= 24:
                score += 2
            else:
                score += 1
            
            # Caffeine scoring
            if row['caffeine_intake'] == 0:
                score += 3
            elif row['caffeine_intake'] == 1:
                score += 2
            else:
                score += 1
            
            # Exercise scoring
            if 30 <= row['exercise_duration'] <= 90:
                score += 3
            elif 15 <= row['exercise_duration'] < 30 or 90 < row['exercise_duration'] <= 120:
                score += 2
            else:
                score += 1
            
            # Screen time scoring
            if row['screen_time_before_bed'] <= 30:
                score += 3
            elif 30 < row['screen_time_before_bed'] <= 60:
                score += 2
            else:
                score += 1
            
            # Stress level scoring
            if row['stress_level'] <= 3:
                score += 3
            elif 3 < row['stress_level'] <= 6:
                score += 2
            else:
                score += 1
            
            # Mood scoring
            if row['mood_before_sleep'] == 0:  # Happy
                score += 3
            elif row['mood_before_sleep'] == 1:  # Neutral
                score += 2
            else:
                score += 1
            
            # Sleep interruptions scoring
            if row['sleep_interruptions'] == 0:
                score += 3
            else:
                score += 1
            
            # Additional factors
            if 18 <= row['room_temperature'] <= 22:
                score += 2
            elif 16 <= row['room_temperature'] < 18 or 22 < row['room_temperature'] <= 24:
                score += 1
            
            if row['noise_level'] <= 3:
                score += 2
            elif 3 < row['noise_level'] <= 6:
                score += 1
            
            if row['light_exposure'] <= 3:
                score += 2
            elif 3 < row['light_exposure'] <= 6:
                score += 1
            
            if row['alcohol_intake'] == 0:
                score += 2
            elif row['alcohol_intake'] == 1:
                score += 1
            
            if 18 <= row['meal_timing'] <= 20:
                score += 2
            elif 17 <= row['meal_timing'] < 18 or 20 < row['meal_timing'] <= 21:
                score += 1
            
            # Classify based on total score
            if score >= 25:
                sleep_quality.append('Good')
            elif score >= 18:
                sleep_quality.append('Average')
            else:
                sleep_quality.append('Poor')
        
        df['sleep_quality'] = sleep_quality
        return df
    
    def load_or_create_models(self):
        """Load existing models or create new ones"""
        if os.path.exists('models/sleep_models.joblib'):
            self.models = joblib.load('models/sleep_models.joblib')
        else:
            self.train_models()
    
    def train_models(self):
        """Train multiple ML models"""
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Generate synthetic data
        df = self.generate_synthetic_data(2000)
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['sleep_quality']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.3f}")
            self.models[name] = model
        
        # Save models
        joblib.dump(self.models, 'models/sleep_models.joblib')
    
    def predict_sleep_quality(self, input_data):
        """Predict sleep quality using ensemble of models"""
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(df)[0]
            prob = model.predict_proba(df)[0]
            predictions[name] = pred
            probabilities[name] = dict(zip(model.classes_, prob))
        
        # Ensemble prediction (majority vote)
        pred_counts = {}
        for pred in predictions.values():
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        final_prediction = max(pred_counts, key=pred_counts.get)
        
        # Calculate average probabilities
        avg_prob = {}
        for class_name in self.models['Random Forest'].classes_:
            avg_prob[class_name] = np.mean([probs.get(class_name, 0) for probs in probabilities.values()])
        
        return final_prediction, avg_prob, predictions
    
    def get_sleep_recommendations(self, input_data, prediction):
        """Generate personalized sleep recommendations"""
        recommendations = []
        
        if input_data['sleep_duration'] < 7:
            recommendations.append("Try to get 7-9 hours of sleep for optimal health")
        elif input_data['sleep_duration'] > 9:
            recommendations.append("Consider if you're oversleeping - 7-9 hours is usually optimal")
        
        if input_data['bedtime_hour'] > 23:
            recommendations.append("Try going to bed before 11 PM for better sleep quality")
        
        if input_data['caffeine_intake'] > 1:
            recommendations.append("Reduce caffeine intake, especially in the afternoon")
        
        if input_data['exercise_duration'] < 30:
            recommendations.append("Aim for at least 30 minutes of exercise daily")
        
        if input_data['screen_time_before_bed'] > 60:
            recommendations.append("Reduce screen time 1 hour before bed")
        
        if input_data['stress_level'] > 6:
            recommendations.append("Try relaxation techniques like meditation or deep breathing")
        
        if input_data['mood_before_sleep'] > 1:
            recommendations.append("Consider journaling or gratitude practice before bed")
        
        if input_data['sleep_interruptions'] == 1:
            recommendations.append("Create a comfortable sleep environment to reduce interruptions")
        
        if input_data['room_temperature'] < 18 or input_data['room_temperature'] > 22:
            recommendations.append("Keep bedroom temperature between 18-22°C")
        
        if input_data['noise_level'] > 6:
            recommendations.append("Use earplugs or white noise to reduce noise")
        
        if input_data['light_exposure'] > 6:
            recommendations.append("Minimize light exposure before bed")
        
        if input_data['alcohol_intake'] > 0:
            recommendations.append("Avoid alcohol close to bedtime")
        
        if input_data['meal_timing'] > 21:
            recommendations.append("Finish eating 2-3 hours before bedtime")
        
        return recommendations

# Initialize predictor
predictor = SleepQualityPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Convert input data
        input_data = {
            'sleep_duration': float(data['sleep_duration']),
            'bedtime_hour': float(data['bedtime_hour']),
            'wakeup_hour': float(data['wakeup_hour']),
            'caffeine_intake': int(data['caffeine_intake']),
            'exercise_duration': float(data['exercise_duration']),
            'screen_time_before_bed': float(data['screen_time_before_bed']),
            'stress_level': float(data['stress_level']),
            'mood_before_sleep': int(data['mood_before_sleep']),
            'sleep_interruptions': int(data['sleep_interruptions']),
            'room_temperature': float(data.get('room_temperature', 20)),
            'noise_level': float(data.get('noise_level', 5)),
            'light_exposure': float(data.get('light_exposure', 5)),
            'alcohol_intake': int(data.get('alcohol_intake', 0)),
            'meal_timing': float(data.get('meal_timing', 19))
        }
        
        # Get prediction
        prediction, probabilities, individual_predictions = predictor.predict_sleep_quality(input_data)
        
        # Get recommendations
        recommendations = predictor.get_sleep_recommendations(input_data, prediction)
        
        return jsonify({
            'prediction': prediction,
            'probabilities': probabilities,
            'individual_predictions': individual_predictions,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
