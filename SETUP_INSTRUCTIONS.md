# Sleep Quality Predictor - Setup Instructions

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run.py
```

### 3. Access the Application
Open your browser and go to: `http://localhost:5000`

## 📋 What's Included

### Core Files
- **`app.py`** - Main Flask application with ML models
- **`run.py`** - Application runner with dependency checking
- **`requirements.txt`** - Python dependencies

### Advanced Features
- **`sleep_analyzer.py`** - Sleep pattern analysis and insights
- **`data_generator.py`** - Realistic synthetic data generation
- **`test_app.py`** - Comprehensive test suite

### Web Interface
- **`templates/index.html`** - Modern, responsive web interface
- **`static/css/style.css`** - Beautiful styling with animations
- **`static/js/script.js`** - Interactive frontend functionality

### Configuration
- **`config.py`** - Application configuration settings
- **`README.md`** - Comprehensive documentation

## 🎯 Key Features Implemented

### ✅ Core ML Functionality
- Multiple ML algorithms (Random Forest, Logistic Regression, SVM, Decision Tree)
- Ensemble prediction with confidence scores
- Real-time sleep quality prediction
- Personalized recommendations engine

### ✅ Advanced Analytics
- Sleep pattern analysis and trend detection
- Factor correlation analysis
- Sleep consistency tracking
- Environmental factor consideration

### ✅ Modern Web Interface
- Responsive design (desktop and mobile)
- Interactive charts and visualizations
- Real-time form validation
- Beautiful UI with sleep-themed design

### ✅ Innovative Features
- Sleep efficiency scoring
- Weekly sleep summary
- Factor impact analysis
- Comprehensive data collection (14+ factors)

## 🔧 Customization Options

### Model Parameters
Edit `app.py` to modify:
- Model hyperparameters
- Feature engineering
- Prediction thresholds
- Recommendation rules

### UI Customization
Edit `static/css/style.css` to change:
- Color scheme
- Layout and spacing
- Animations and transitions
- Responsive breakpoints

### Data Generation
Edit `data_generator.py` to adjust:
- Dataset size and complexity
- Factor correlations
- Demographic variations
- Temporal patterns

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_app.py
```

## 📊 Sample Data

The application automatically generates realistic sleep data with:
- Correlated factors and realistic patterns
- Demographic variations (age, gender, occupation)
- Environmental factors (temperature, noise, light)
- Temporal patterns and seasonal variations

## 🌟 Usage Tips

1. **Input Data**: Fill in all fields for best predictions
2. **Consistency**: Regular data entry improves pattern analysis
3. **Recommendations**: Follow personalized tips for better sleep
4. **Trends**: Monitor your sleep patterns over time
5. **Factors**: Pay attention to factor impact analysis

## 🚀 Deployment

### Local Development
```bash
python run.py
```

### Production Deployment
1. Set environment variables:
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secret-key
   ```

2. Use a production WSGI server:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## 🔍 Troubleshooting

### Common Issues
1. **Port already in use**: Change port in `run.py`
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Model not found**: Models are auto-generated on first run
4. **Browser compatibility**: Use modern browsers (Chrome, Firefox, Safari)

### Performance Optimization
- Increase `DATASET_SIZE` in `config.py` for better model accuracy
- Use production configuration for better performance
- Consider model caching for high-traffic scenarios

## 📈 Next Steps

1. **Data Collection**: Start collecting real sleep data
2. **Model Training**: Retrain models with your data
3. **Feature Engineering**: Add new factors based on insights
4. **Mobile App**: Develop native mobile applications
5. **API Integration**: Connect with wearable devices

## 🎉 Success!

Your Sleep Quality Predictor is now ready to use! The application provides:
- Accurate sleep quality predictions
- Personalized improvement recommendations
- Beautiful, intuitive interface
- Comprehensive sleep analytics
- Extensible architecture for future enhancements

Enjoy better sleep! 🌙✨
