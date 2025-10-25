# Sleep Quality Predictor 🌙

A comprehensive Machine Learning-based Sleep Quality Predictor that analyzes lifestyle, behavioral, and physiological factors to predict and improve sleep quality. This project combines advanced ML algorithms with an intuitive web interface to provide personalized sleep insights and recommendations.

## ✨ Features

### Core Functionality
- **Sleep Quality Prediction**: Predicts sleep quality (Good/Average/Poor) based on multiple factors
- **Multiple ML Algorithms**: Uses Random Forest, Logistic Regression, SVM, and Decision Tree
- **Real-time Analysis**: Instant predictions with confidence scores
- **Personalized Recommendations**: AI-generated tips for improving sleep quality

### Advanced Features
- **Sleep Pattern Analysis**: Tracks sleep trends and consistency over time
- **Factor Impact Analysis**: Identifies which factors most affect your sleep
- **Sleep Efficiency Scoring**: Calculates overall sleep efficiency percentage
- **Interactive Dashboard**: Visual analytics with charts and trends
- **Comprehensive Data Collection**: 14+ factors including environmental and lifestyle data

### Innovative Additions
- **Sleep Consistency Tracking**: Monitors bedtime and wake-up time consistency
- **Correlation Analysis**: Shows relationships between different sleep factors
- **Trend Analysis**: Identifies improving, declining, or stable sleep patterns
- **Environmental Factors**: Considers room temperature, noise, and light exposure
- **Lifestyle Integration**: Analyzes exercise, caffeine, alcohol, and meal timing
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sleep-quality-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## 📊 Data Input Factors

The predictor analyzes the following factors:

### Basic Sleep Data
- **Sleep Duration**: Hours of sleep (3-12 hours)
- **Bedtime**: Time you go to bed
- **Wake-up Time**: Time you wake up

### Lifestyle Factors
- **Caffeine Intake**: None/Low/Moderate/High
- **Exercise Duration**: Minutes of daily exercise
- **Screen Time Before Bed**: Minutes of screen usage before sleep
- **Stress Level**: Self-rated stress (0-10 scale)
- **Mood Before Sleep**: Happy/Neutral/Sad/Anxious
- **Sleep Interruptions**: Whether you wake up during the night

### Environmental Factors
- **Room Temperature**: Bedroom temperature in Celsius
- **Noise Level**: Environmental noise level (0-10 scale)
- **Light Exposure**: Light exposure before bed (0-10 scale)
- **Alcohol Intake**: Alcohol consumption level
- **Meal Timing**: Time of last meal

## 🤖 Machine Learning Models

The system uses an ensemble approach with multiple algorithms:

1. **Random Forest**: Primary model for robust predictions
2. **Logistic Regression**: Linear relationship analysis
3. **Support Vector Machine (SVM)**: Non-linear pattern recognition
4. **Decision Tree**: Interpretable rule-based predictions

### Model Training
- **Dataset**: Synthetic sleep data with realistic correlations
- **Features**: 14 input features with engineered relationships
- **Target**: 3-class sleep quality classification
- **Validation**: Cross-validation with 80/20 train-test split

## 📈 Analytics Dashboard

### Sleep Quality Trends
- Weekly sleep quality scores
- Trend analysis (improving/declining/stable)
- Consistency metrics

### Factor Impact Analysis
- Correlation between factors and sleep quality
- Feature importance ranking
- Personalized factor recommendations

### Sleep Efficiency Score
- Overall sleep efficiency percentage
- Based on duration, consistency, and quality
- Historical tracking and improvement suggestions

## 🛠️ Technical Architecture

### Backend (Flask)
- **Framework**: Flask web framework
- **ML Libraries**: scikit-learn, pandas, numpy
- **Data Processing**: Real-time feature engineering
- **API Endpoints**: RESTful API for predictions

### Frontend (HTML/CSS/JavaScript)
- **Responsive Design**: Mobile-first approach
- **Interactive Charts**: Chart.js for data visualization
- **Modern UI**: Clean, intuitive interface
- **Real-time Updates**: Dynamic form validation and feedback

### Data Pipeline
- **Data Generation**: Realistic synthetic dataset creation
- **Feature Engineering**: Advanced correlation modeling
- **Model Training**: Automated model training and validation
- **Prediction Pipeline**: Real-time prediction processing

## 📁 Project Structure

```
sleep-quality-predictor/
├── app.py                 # Main Flask application
├── sleep_analyzer.py      # Advanced sleep pattern analysis
├── data_generator.py      # Synthetic data generation
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── css/
│   │   └── style.css     # Styling and responsive design
│   ├── js/
│   │   └── script.js     # Frontend functionality
│   └── images/           # Static assets
└── models/               # Trained ML models (auto-generated)
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file for custom configuration:
```env
FLASK_ENV=development
MODEL_PATH=models/
DATASET_SIZE=1000
```

### Model Customization
Modify `app.py` to adjust:
- Model parameters
- Feature engineering
- Prediction thresholds
- Recommendation rules

## 📊 Sample Data

The application includes a comprehensive dataset generator that creates realistic sleep data with:
- **Correlated Factors**: Realistic relationships between sleep factors
- **Temporal Patterns**: Seasonal and weekly variations
- **Demographic Variation**: Age and occupation-based patterns
- **Environmental Simulation**: Urban vs rural sleep environments

## 🎯 Use Cases

### Personal Sleep Tracking
- Daily sleep quality monitoring
- Lifestyle factor analysis
- Sleep improvement recommendations
- Progress tracking over time

### Healthcare Applications
- Sleep disorder screening
- Treatment effectiveness monitoring
- Patient sleep pattern analysis
- Clinical decision support

### Research and Analytics
- Sleep pattern research
- Population sleep health studies
- Factor correlation analysis
- Sleep intervention effectiveness

## 🚀 Future Enhancements

### Planned Features
- **Mobile App**: Native iOS and Android applications
- **Wearable Integration**: Smartwatch and fitness tracker data
- **Sleep Diary**: Text analysis of sleep journal entries
- **Social Features**: Sleep challenges and community support
- **AI Chatbot**: Personalized sleep coaching assistant

### Advanced Analytics
- **Deep Learning**: Neural networks for complex pattern recognition
- **Time Series Analysis**: Advanced temporal pattern detection
- **Clustering**: Sleep pattern grouping and classification
- **Anomaly Detection**: Unusual sleep pattern identification

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset Sources**: Sleep Health and Lifestyle Dataset (Kaggle), Sleep Efficiency Dataset (UCI ML Repo)
- **Libraries**: scikit-learn, Flask, Chart.js, Font Awesome
- **Design Inspiration**: Modern sleep tracking applications and health analytics platforms
## 🖥️ Output

![Sleep Quality Predictor Output](D:\Sleep Quality predictor\image.png)
![Sleep Quality Predictor Output](D:\Sleep Quality predictor\image-1.png)
![Sleep Quality Predictor Output](D:\Sleep Quality predictor\image-2.png)
![Sleep Quality Predictor Output](D:\Sleep Quality predictor\image-3.png)
![Sleep Quality Predictor Output](D:\Sleep Quality predictor\image-4.png)


**Sleep Quality Predictor** - Transforming sleep health through AI and machine learning. 🌙✨

*Built with ❤️ for better sleep and healthier lives.*


