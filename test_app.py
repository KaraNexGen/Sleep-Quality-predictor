"""
Test Suite for Sleep Quality Predictor
Comprehensive testing for all components
"""

import unittest
import json
import sys
import os
from unittest.mock import patch, MagicMock

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, SleepQualityPredictor
from sleep_analyzer import SleepPatternAnalyzer, SleepRecommendationEngine
from data_generator import SleepDataGenerator

class TestSleepQualityPredictor(unittest.TestCase):
    """Test cases for the main SleepQualityPredictor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.predictor = SleepQualityPredictor()
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        self.assertIsNotNone(self.predictor)
        self.assertIsNotNone(self.predictor.models)
        self.assertIsNotNone(self.predictor.feature_names)
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        data = self.predictor.generate_synthetic_data(100)
        self.assertEqual(len(data), 100)
        self.assertIn('sleep_quality', data.columns)
        self.assertIn('sleep_duration', data.columns)
    
    def test_prediction_with_valid_data(self):
        """Test prediction with valid input data"""
        input_data = {
            'sleep_duration': 7.5,
            'bedtime_hour': 22.5,
            'wakeup_hour': 6.0,
            'caffeine_intake': 1,
            'exercise_duration': 30,
            'screen_time_before_bed': 30,
            'stress_level': 5,
            'mood_before_sleep': 0,
            'sleep_interruptions': 0,
            'room_temperature': 20,
            'noise_level': 3,
            'light_exposure': 3,
            'alcohol_intake': 0,
            'meal_timing': 19
        }
        
        prediction, probabilities, individual_predictions = self.predictor.predict_sleep_quality(input_data)
        
        self.assertIn(prediction, ['Good', 'Average', 'Poor'])
        self.assertIsInstance(probabilities, dict)
        self.assertIsInstance(individual_predictions, dict)
    
    def test_recommendations_generation(self):
        """Test recommendations generation"""
        input_data = {
            'sleep_duration': 6.0,
            'bedtime_hour': 23.5,
            'wakeup_hour': 6.0,
            'caffeine_intake': 3,
            'exercise_duration': 15,
            'screen_time_before_bed': 120,
            'stress_level': 8,
            'mood_before_sleep': 2,
            'sleep_interruptions': 1,
            'room_temperature': 25,
            'noise_level': 7,
            'light_exposure': 8,
            'alcohol_intake': 2,
            'meal_timing': 21
        }
        
        recommendations = self.predictor.get_sleep_recommendations(input_data, 'Poor')
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

class TestSleepPatternAnalyzer(unittest.TestCase):
    """Test cases for SleepPatternAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SleepPatternAnalyzer()
    
    def test_add_sleep_record(self):
        """Test adding sleep records"""
        sleep_data = {
            'sleep_duration': 7.5,
            'bedtime_hour': 22.5,
            'wakeup_hour': 6.0,
            'sleep_quality': 'Good',
            'exercise_duration': 45,
            'screen_time_before_bed': 30,
            'stress_level': 4
        }
        
        self.analyzer.add_sleep_record(sleep_data)
        self.assertEqual(len(self.analyzer.sleep_history), 1)
    
    def test_sleep_pattern_analysis(self):
        """Test sleep pattern analysis"""
        # Add multiple sleep records
        for i in range(5):
            sleep_data = {
                'sleep_duration': 7.5 + i * 0.1,
                'bedtime_hour': 22.5,
                'wakeup_hour': 6.0,
                'sleep_quality': 'Good' if i < 3 else 'Average',
                'exercise_duration': 45,
                'screen_time_before_bed': 30,
                'stress_level': 4
            }
            self.analyzer.add_sleep_record(sleep_data)
        
        insights = self.analyzer.analyze_sleep_patterns()
        self.assertIsInstance(insights, dict)
        self.assertIn('average_sleep_duration', insights)
        self.assertIn('sleep_consistency', insights)
        self.assertIn('recommendations', insights)

class TestSleepRecommendationEngine(unittest.TestCase):
    """Test cases for SleepRecommendationEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = SleepRecommendationEngine()
    
    def test_recommendations_generation(self):
        """Test recommendations generation"""
        sleep_data = {
            'sleep_duration': 6.0,
            'bedtime': 23.5,
            'exercise': 10,
            'screen_time': 120,
            'stress': 8
        }
        
        recommendations = self.engine.get_recommendations(sleep_data)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check recommendation structure
        for rec in recommendations:
            self.assertIn('factor', rec)
            self.assertIn('message', rec)
            self.assertIn('priority', rec)

class TestDataGenerator(unittest.TestCase):
    """Test cases for SleepDataGenerator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = SleepDataGenerator(seed=42)
    
    def test_realistic_dataset_generation(self):
        """Test realistic dataset generation"""
        df = self.generator.generate_realistic_dataset(100)
        
        self.assertEqual(len(df), 100)
        self.assertIn('sleep_quality', df.columns)
        self.assertIn('sleep_duration', df.columns)
        self.assertIn('user_id', df.columns)
        
        # Check sleep quality distribution
        quality_counts = df['sleep_quality'].value_counts()
        self.assertGreater(quality_counts.sum(), 0)
    
    def test_user_specific_data_generation(self):
        """Test user-specific data generation"""
        user_profile = {
            'user_id': 'test_user',
            'age': 30,
            'gender': 'Female',
            'occupation': 'Office Worker',
            'preferred_sleep_duration': 8.0,
            'preferred_bedtime': 23.0
        }
        
        df = self.generator.generate_user_specific_data(user_profile, 7)
        
        self.assertEqual(len(df), 7)
        self.assertTrue(all(df['user_id'] == 'test_user'))
        self.assertIn('sleep_quality', df.columns)

class TestFlaskAPI(unittest.TestCase):
    """Test cases for Flask API endpoints"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
    
    def test_home_page(self):
        """Test home page accessibility"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_prediction_endpoint(self):
        """Test prediction endpoint with valid data"""
        test_data = {
            'sleep_duration': 7.5,
            'bedtime_hour': 22.5,
            'wakeup_hour': 6.0,
            'caffeine_intake': 1,
            'exercise_duration': 30,
            'screen_time_before_bed': 30,
            'stress_level': 5,
            'mood_before_sleep': 0,
            'sleep_interruptions': 0,
            'room_temperature': 20,
            'noise_level': 3,
            'light_exposure': 3,
            'alcohol_intake': 0,
            'meal_timing': 19
        }
        
        response = self.app.post('/predict', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('probabilities', data)
        self.assertIn('recommendations', data)
    
    def test_prediction_endpoint_invalid_data(self):
        """Test prediction endpoint with invalid data"""
        test_data = {
            'sleep_duration': 'invalid',
            'bedtime_hour': 22.5
        }
        
        response = self.app.post('/predict',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_prediction(self):
        """Test complete prediction workflow"""
        # Initialize components
        predictor = SleepQualityPredictor()
        analyzer = SleepPatternAnalyzer()
        
        # Generate test data
        generator = SleepDataGenerator()
        df = generator.generate_realistic_dataset(10)
        
        # Test prediction for each record
        for _, row in df.iterrows():
            input_data = row.to_dict()
            prediction, probabilities, individual_predictions = predictor.predict_sleep_quality(input_data)
            
            self.assertIn(prediction, ['Good', 'Average', 'Poor'])
            self.assertIsInstance(probabilities, dict)
            self.assertIsInstance(individual_predictions, dict)
            
            # Add to analyzer
            analyzer.add_sleep_record(input_data)
        
        # Test pattern analysis
        insights = analyzer.analyze_sleep_patterns()
        self.assertIsInstance(insights, dict)
        self.assertIn('sleep_score', insights)

def run_tests():
    """Run all tests"""
    print("🧪 Running Sleep Quality Predictor Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSleepQualityPredictor,
        TestSleepPatternAnalyzer,
        TestSleepRecommendationEngine,
        TestDataGenerator,
        TestFlaskAPI,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Test Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Test Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
