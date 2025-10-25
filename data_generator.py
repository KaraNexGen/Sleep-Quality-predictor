"""
Advanced Sleep Data Generator
Generates realistic sleep datasets with various patterns and correlations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import json

class SleepDataGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
    
    def generate_realistic_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate a realistic sleep dataset with correlations and patterns"""
        
        data = {
            'user_id': [f"user_{i:04d}" for i in range(n_samples)],
            'age': np.random.normal(35, 12, n_samples).clip(18, 80).astype(int),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
            'occupation': np.random.choice([
                'Student', 'Office Worker', 'Healthcare Worker', 'Shift Worker', 
                'Remote Worker', 'Retired', 'Unemployed'
            ], n_samples, p=[0.15, 0.25, 0.15, 0.1, 0.15, 0.15, 0.05]),
        }
        
        # Generate correlated sleep data
        sleep_data = self._generate_correlated_sleep_data(n_samples)
        data.update(sleep_data)
        
        # Add lifestyle factors
        lifestyle_data = self._generate_lifestyle_data(n_samples, data['age'])
        data.update(lifestyle_data)
        
        # Add environmental factors
        env_data = self._generate_environmental_data(n_samples)
        data.update(env_data)
        
        # Generate sleep quality based on all factors
        data['sleep_quality'] = self._calculate_sleep_quality(data)
        
        # Add timestamps
        data['date'] = self._generate_dates(n_samples)
        
        return pd.DataFrame(data)
    
    def _generate_correlated_sleep_data(self, n_samples: int) -> Dict:
        """Generate sleep data with realistic correlations"""
        
        # Base sleep duration (7-9 hours is optimal)
        base_duration = np.random.normal(7.5, 1.2, n_samples)
        
        # Bedtime affects sleep duration
        bedtime_hour = np.random.normal(22.5, 1.8, n_samples)
        bedtime_effect = (bedtime_hour - 22.5) * 0.3  # Later bedtime = less sleep
        
        # Stress affects sleep duration
        stress_level = np.random.exponential(3, n_samples).clip(0, 10)
        stress_effect = -stress_level * 0.1
        
        # Exercise affects sleep duration (moderate exercise helps)
        exercise_duration = np.random.exponential(25, n_samples).clip(0, 180)
        exercise_effect = np.where(
            exercise_duration < 15, -0.2,  # Too little exercise
            np.where(exercise_duration > 90, -0.1, 0.1)  # Too much or just right
        )
        
        # Calculate final sleep duration
        sleep_duration = (base_duration + bedtime_effect + stress_effect + exercise_effect).clip(3, 12)
        
        # Wake-up time based on bedtime and duration
        wakeup_hour = (bedtime_hour + sleep_duration) % 24
        
        return {
            'sleep_duration': sleep_duration,
            'bedtime_hour': bedtime_hour,
            'wakeup_hour': wakeup_hour,
            'stress_level': stress_level,
            'exercise_duration': exercise_duration
        }
    
    def _generate_lifestyle_data(self, n_samples: int, ages: np.ndarray) -> Dict:
        """Generate lifestyle factors correlated with age and other factors"""
        
        # Caffeine intake (higher in younger adults)
        caffeine_base = np.where(ages < 30, 2.5, np.where(ages < 50, 2.0, 1.5))
        caffeine_intake = np.random.poisson(caffeine_base, n_samples).clip(0, 3)
        
        # Screen time (higher in younger people)
        screen_base = np.where(ages < 30, 120, np.where(ages < 50, 90, 60))
        screen_time_before_bed = np.random.exponential(screen_base / 60, n_samples).clip(0, 300)
        
        # Alcohol intake (moderate in middle age)
        alcohol_base = np.where(ages < 25, 0.5, np.where(ages < 45, 1.5, 1.0))
        alcohol_intake = np.random.poisson(alcohol_base, n_samples).clip(0, 3)
        
        # Meal timing (earlier in older adults)
        meal_base = np.where(ages < 30, 20.5, np.where(ages < 50, 19.5, 18.5))
        meal_timing = np.random.normal(meal_base, 1.5, n_samples).clip(16, 23)
        
        # Mood (correlated with stress and exercise)
        mood_scores = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1])
        
        return {
            'caffeine_intake': caffeine_intake,
            'screen_time_before_bed': screen_time_before_bed,
            'alcohol_intake': alcohol_intake,
            'meal_timing': meal_timing,
            'mood_before_sleep': mood_scores
        }
    
    def _generate_environmental_data(self, n_samples: int) -> Dict:
        """Generate environmental factors"""
        
        # Room temperature (seasonal variation)
        base_temp = np.random.normal(20, 2, n_samples)
        seasonal_variation = np.sin(np.random.uniform(0, 2*np.pi, n_samples)) * 3
        room_temperature = (base_temp + seasonal_variation).clip(15, 30)
        
        # Noise level (urban vs rural simulation)
        urban_factor = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        noise_level = np.random.exponential(3, n_samples) + urban_factor * 2
        noise_level = noise_level.clip(0, 10)
        
        # Light exposure (correlated with screen time)
        light_exposure = np.random.exponential(4, n_samples).clip(0, 10)
        
        # Sleep interruptions (correlated with stress and environment)
        interruption_prob = 0.2 + (noise_level / 10) * 0.3
        sleep_interruptions = np.random.binomial(1, interruption_prob, n_samples)
        
        return {
            'room_temperature': room_temperature,
            'noise_level': noise_level,
            'light_exposure': light_exposure,
            'sleep_interruptions': sleep_interruptions
        }
    
    def _calculate_sleep_quality(self, data: Dict) -> List[str]:
        """Calculate sleep quality based on all factors"""
        
        scores = []
        
        for i in range(len(data['sleep_duration'])):
            score = 0
            
            # Sleep duration scoring
            duration = data['sleep_duration'][i]
            if 7 <= duration <= 9:
                score += 3
            elif 6 <= duration < 7 or 9 < duration <= 10:
                score += 2
            else:
                score += 1
            
            # Bedtime scoring
            bedtime = data['bedtime_hour'][i]
            if 22 <= bedtime <= 23:
                score += 3
            elif 21 <= bedtime < 22 or 23 < bedtime <= 24:
                score += 2
            else:
                score += 1
            
            # Exercise scoring
            exercise = data['exercise_duration'][i]
            if 30 <= exercise <= 90:
                score += 3
            elif 15 <= exercise < 30 or 90 < exercise <= 120:
                score += 2
            else:
                score += 1
            
            # Screen time scoring
            screen_time = data['screen_time_before_bed'][i]
            if screen_time <= 30:
                score += 3
            elif 30 < screen_time <= 60:
                score += 2
            else:
                score += 1
            
            # Stress scoring
            stress = data['stress_level'][i]
            if stress <= 3:
                score += 3
            elif 3 < stress <= 6:
                score += 2
            else:
                score += 1
            
            # Caffeine scoring
            caffeine = data['caffeine_intake'][i]
            if caffeine == 0:
                score += 3
            elif caffeine == 1:
                score += 2
            else:
                score += 1
            
            # Mood scoring
            mood = data['mood_before_sleep'][i]
            if mood == 0:  # Happy
                score += 3
            elif mood == 1:  # Neutral
                score += 2
            else:
                score += 1
            
            # Sleep interruptions
            interruptions = data['sleep_interruptions'][i]
            if interruptions == 0:
                score += 3
            else:
                score += 1
            
            # Environmental factors
            temp = data['room_temperature'][i]
            if 18 <= temp <= 22:
                score += 2
            elif 16 <= temp < 18 or 22 < temp <= 24:
                score += 1
            
            noise = data['noise_level'][i]
            if noise <= 3:
                score += 2
            elif 3 < noise <= 6:
                score += 1
            
            light = data['light_exposure'][i]
            if light <= 3:
                score += 2
            elif 3 < light <= 6:
                score += 1
            
            alcohol = data['alcohol_intake'][i]
            if alcohol == 0:
                score += 2
            elif alcohol == 1:
                score += 1
            
            meal = data['meal_timing'][i]
            if 18 <= meal <= 20:
                score += 2
            elif 17 <= meal < 18 or 20 < meal <= 21:
                score += 1
            
            # Classify sleep quality
            if score >= 25:
                scores.append('Good')
            elif score >= 18:
                scores.append('Average')
            else:
                scores.append('Poor')
        
        return scores
    
    def _generate_dates(self, n_samples: int) -> List[str]:
        """Generate realistic dates for the dataset"""
        start_date = datetime.now() - timedelta(days=30)
        dates = []
        
        for i in range(n_samples):
            random_days = random.randint(0, 30)
            date = start_date + timedelta(days=random_days)
            dates.append(date.strftime('%Y-%m-%d'))
        
        return dates
    
    def generate_user_specific_data(self, user_profile: Dict, n_days: int = 30) -> pd.DataFrame:
        """Generate data for a specific user profile"""
        
        data = {
            'user_id': [user_profile.get('user_id', 'user_0001')] * n_days,
            'age': [user_profile.get('age', 30)] * n_days,
            'gender': [user_profile.get('gender', 'Other')] * n_days,
            'occupation': [user_profile.get('occupation', 'Office Worker')] * n_days,
        }
        
        # Generate data based on user profile
        base_duration = user_profile.get('preferred_sleep_duration', 7.5)
        bedtime_preference = user_profile.get('preferred_bedtime', 22.5)
        
        sleep_duration = np.random.normal(base_duration, 0.8, n_days).clip(4, 12)
        bedtime_hour = np.random.normal(bedtime_preference, 1.0, n_days)
        wakeup_hour = (bedtime_hour + sleep_duration) % 24
        
        data.update({
            'sleep_duration': sleep_duration,
            'bedtime_hour': bedtime_hour,
            'wakeup_hour': wakeup_hour,
            'stress_level': np.random.exponential(3, n_days).clip(0, 10),
            'exercise_duration': np.random.exponential(30, n_days).clip(0, 180),
            'caffeine_intake': np.random.choice([0, 1, 2, 3], n_days, p=[0.2, 0.3, 0.3, 0.2]),
            'screen_time_before_bed': np.random.exponential(60, n_days).clip(0, 300),
            'mood_before_sleep': np.random.choice([0, 1, 2, 3], n_days, p=[0.3, 0.4, 0.2, 0.1]),
            'sleep_interruptions': np.random.choice([0, 1], n_days, p=[0.7, 0.3]),
            'room_temperature': np.random.normal(20, 2, n_days).clip(15, 30),
            'noise_level': np.random.uniform(0, 10, n_days),
            'light_exposure': np.random.uniform(0, 10, n_days),
            'alcohol_intake': np.random.choice([0, 1, 2, 3], n_days, p=[0.6, 0.2, 0.15, 0.05]),
            'meal_timing': np.random.normal(19, 2, n_days).clip(16, 23)
        })
        
        # Calculate sleep quality
        data['sleep_quality'] = self._calculate_sleep_quality(data)
        
        # Generate dates
        data['date'] = self._generate_dates(n_days)
        
        return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    generator = SleepDataGenerator()
    
    # Generate a large dataset
    print("Generating realistic sleep dataset...")
    df = generator.generate_realistic_dataset(1000)
    
    print(f"Generated dataset with {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"Sleep quality distribution:")
    print(df['sleep_quality'].value_counts())
    
    # Save to CSV
    df.to_csv('sleep_dataset.csv', index=False)
    print("Dataset saved to 'sleep_dataset.csv'")
    
    # Generate user-specific data
    user_profile = {
        'user_id': 'user_1234',
        'age': 28,
        'gender': 'Female',
        'occupation': 'Software Developer',
        'preferred_sleep_duration': 8.0,
        'preferred_bedtime': 23.0
    }
    
    user_df = generator.generate_user_specific_data(user_profile, 14)
    print(f"\nGenerated {len(user_df)} days of data for user {user_profile['user_id']}")
    print(user_df[['date', 'sleep_duration', 'sleep_quality']].head())
