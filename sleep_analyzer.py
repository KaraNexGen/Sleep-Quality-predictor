"""
Advanced Sleep Pattern Analyzer
Provides detailed sleep analysis and insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SleepPatternAnalyzer:
    def __init__(self):
        self.sleep_history = []
        self.insights = {}
    
    def add_sleep_record(self, sleep_data: Dict) -> None:
        """Add a new sleep record to the history"""
        sleep_data['timestamp'] = datetime.now()
        self.sleep_history.append(sleep_data)
    
    def analyze_sleep_patterns(self) -> Dict:
        """Analyze sleep patterns and generate insights"""
        if not self.sleep_history:
            return {"error": "No sleep data available"}
        
        df = pd.DataFrame(self.sleep_history)
        
        insights = {
            'average_sleep_duration': df['sleep_duration'].mean(),
            'sleep_consistency': self.calculate_sleep_consistency(df),
            'bedtime_consistency': self.calculate_bedtime_consistency(df),
            'wakeup_consistency': self.calculate_wakeup_consistency(df),
            'sleep_quality_trend': self.calculate_sleep_quality_trend(df),
            'factor_correlations': self.calculate_factor_correlations(df),
            'recommendations': self.generate_personalized_recommendations(df),
            'sleep_score': self.calculate_overall_sleep_score(df)
        }
        
        return insights
    
    def calculate_sleep_consistency(self, df: pd.DataFrame) -> float:
        """Calculate sleep duration consistency (lower std = more consistent)"""
        return 1 / (1 + df['sleep_duration'].std())
    
    def calculate_bedtime_consistency(self, df: pd.DataFrame) -> float:
        """Calculate bedtime consistency"""
        bedtime_std = df['bedtime_hour'].std()
        return 1 / (1 + bedtime_std)
    
    def calculate_wakeup_consistency(self, df: pd.DataFrame) -> float:
        """Calculate wake-up time consistency"""
        wakeup_std = df['wakeup_hour'].std()
        return 1 / (1 + wakeup_std)
    
    def calculate_sleep_quality_trend(self, df: pd.DataFrame) -> str:
        """Calculate sleep quality trend over time"""
        if len(df) < 3:
            return "Insufficient data"
        
        # Convert sleep quality to numeric scores
        quality_scores = df['sleep_quality'].map({'Poor': 1, 'Average': 2, 'Good': 3})
        
        # Calculate trend
        x = np.arange(len(quality_scores))
        slope = np.polyfit(x, quality_scores, 1)[0]
        
        if slope > 0.1:
            return "Improving"
        elif slope < -0.1:
            return "Declining"
        else:
            return "Stable"
    
    def calculate_factor_correlations(self, df: pd.DataFrame) -> Dict:
        """Calculate correlations between factors and sleep quality"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_columns:
            if col != 'sleep_duration':  # Avoid self-correlation
                corr = df[col].corr(df['sleep_duration'])
                correlations[col] = corr
        
        return correlations
    
    def generate_personalized_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate personalized recommendations based on patterns"""
        recommendations = []
        
        # Sleep duration analysis
        avg_duration = df['sleep_duration'].mean()
        if avg_duration < 7:
            recommendations.append("Your average sleep duration is below recommended levels. Try to get 7-9 hours of sleep.")
        elif avg_duration > 9:
            recommendations.append("You might be oversleeping. Consider if 7-9 hours would be more optimal.")
        
        # Bedtime consistency
        bedtime_std = df['bedtime_hour'].std()
        if bedtime_std > 1.5:
            recommendations.append("Your bedtime varies significantly. Try to maintain a consistent bedtime.")
        
        # Exercise correlation
        exercise_corr = df['exercise_duration'].corr(df['sleep_duration'])
        if exercise_corr < 0.3:
            recommendations.append("Regular exercise might improve your sleep quality.")
        
        # Screen time analysis
        avg_screen_time = df['screen_time_before_bed'].mean()
        if avg_screen_time > 60:
            recommendations.append("Consider reducing screen time before bed to improve sleep quality.")
        
        # Stress analysis
        avg_stress = df['stress_level'].mean()
        if avg_stress > 6:
            recommendations.append("High stress levels may be affecting your sleep. Consider stress management techniques.")
        
        return recommendations
    
    def calculate_overall_sleep_score(self, df: pd.DataFrame) -> float:
        """Calculate overall sleep score (0-100)"""
        # Convert sleep quality to numeric
        quality_scores = df['sleep_quality'].map({'Poor': 1, 'Average': 2, 'Good': 3})
        
        # Calculate weighted score
        duration_score = min(100, (df['sleep_duration'].mean() / 9) * 100)
        consistency_score = self.calculate_sleep_consistency(df) * 100
        quality_score = quality_scores.mean() / 3 * 100
        
        overall_score = (duration_score * 0.4 + consistency_score * 0.3 + quality_score * 0.3)
        return round(overall_score, 1)
    
    def generate_sleep_report(self) -> str:
        """Generate a comprehensive sleep report"""
        insights = self.analyze_sleep_patterns()
        
        report = f"""
        SLEEP QUALITY ANALYSIS REPORT
        =============================
        
        Overall Sleep Score: {insights.get('sleep_score', 'N/A')}/100
        
        AVERAGE METRICS:
        - Sleep Duration: {insights.get('average_sleep_duration', 'N/A'):.1f} hours
        - Sleep Consistency: {insights.get('sleep_consistency', 'N/A'):.2f}
        - Bedtime Consistency: {insights.get('bedtime_consistency', 'N/A'):.2f}
        - Wake-up Consistency: {insights.get('wakeup_consistency', 'N/A'):.2f}
        
        TREND ANALYSIS:
        - Sleep Quality Trend: {insights.get('sleep_quality_trend', 'N/A')}
        
        PERSONALIZED RECOMMENDATIONS:
        """
        
        for i, rec in enumerate(insights.get('recommendations', []), 1):
            report += f"\n{i}. {rec}"
        
        return report

class SleepRecommendationEngine:
    def __init__(self):
        self.recommendation_rules = {
            'sleep_duration': {
                'low': (0, 6.5, "Try to get at least 7 hours of sleep for optimal health"),
                'optimal': (7, 9, "Great! You're getting the recommended amount of sleep"),
                'high': (9, 12, "Consider if you might be oversleeping")
            },
            'bedtime': {
                'early': (18, 21, "Very early bedtime - ensure you're getting enough sleep"),
                'optimal': (21, 23, "Good bedtime! This aligns with natural circadian rhythms"),
                'late': (23, 26, "Try going to bed earlier for better sleep quality")
            },
            'exercise': {
                'low': (0, 15, "Regular exercise can significantly improve sleep quality"),
                'moderate': (15, 60, "Good exercise routine! This should help your sleep"),
                'high': (60, 300, "Excellent exercise! Just avoid intense workouts close to bedtime")
            },
            'screen_time': {
                'low': (0, 30, "Excellent! Minimal screen time before bed"),
                'moderate': (30, 90, "Consider reducing screen time by 30 minutes"),
                'high': (90, 300, "High screen time can disrupt sleep - try reducing by 1 hour")
            },
            'stress': {
                'low': (0, 3, "Low stress levels - great for sleep!"),
                'moderate': (3, 7, "Moderate stress - consider relaxation techniques"),
                'high': (7, 10, "High stress levels - try meditation or deep breathing")
            }
        }
    
    def get_recommendations(self, sleep_data: Dict) -> List[Dict]:
        """Get personalized recommendations based on sleep data"""
        recommendations = []
        
        for factor, rules in self.recommendation_rules.items():
            if factor in sleep_data:
                value = sleep_data[factor]
                
                for category, (min_val, max_val, message) in rules.items():
                    if min_val <= value < max_val:
                        recommendations.append({
                            'factor': factor,
                            'category': category,
                            'message': message,
                            'priority': self.get_priority(factor, category)
                        })
                        break
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        return recommendations
    
    def get_priority(self, factor: str, category: str) -> int:
        """Get priority score for recommendation"""
        priority_scores = {
            'sleep_duration': {'low': 10, 'optimal': 1, 'high': 8},
            'bedtime': {'early': 6, 'optimal': 1, 'late': 9},
            'exercise': {'low': 8, 'moderate': 3, 'high': 2},
            'screen_time': {'low': 1, 'moderate': 6, 'high': 9},
            'stress': {'low': 1, 'moderate': 7, 'high': 10}
        }
        
        return priority_scores.get(factor, {}).get(category, 5)

# Example usage and testing
if __name__ == "__main__":
    # Test the analyzer
    analyzer = SleepPatternAnalyzer()
    
    # Add some sample data
    sample_data = [
        {
            'sleep_duration': 7.5,
            'bedtime_hour': 22.5,
            'wakeup_hour': 6.0,
            'sleep_quality': 'Good',
            'exercise_duration': 45,
            'screen_time_before_bed': 30,
            'stress_level': 4
        },
        {
            'sleep_duration': 6.5,
            'bedtime_hour': 23.5,
            'wakeup_hour': 6.0,
            'sleep_quality': 'Average',
            'exercise_duration': 20,
            'screen_time_before_bed': 90,
            'stress_level': 7
        }
    ]
    
    for data in sample_data:
        analyzer.add_sleep_record(data)
    
    # Generate insights
    insights = analyzer.analyze_sleep_patterns()
    print("Sleep Analysis Insights:")
    print(json.dumps(insights, indent=2))
    
    # Generate report
    report = analyzer.generate_sleep_report()
    print("\nSleep Report:")
    print(report)
