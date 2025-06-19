# 🎮 DeepKick

DeepKick is an advanced FIFA player prediction web application that uses machine learning to analyze player attributes and generate personalized FIFA-style player cards. The application provides detailed player comparisons with real-world professional footballers and suggests optimal playing positions based on the input attributes.

## ✨ Features

### 🎯 Player Prediction
- Predicts dominant foot (Left/Right) using machine learning
- Generates FIFA-style player cards with custom attributes
- Multiple card template options (Gold, TOTY, Silver, Bronze)
- Custom photo upload support

### 📊 Advanced Position Analysis
- Determines optimal playing position based on attributes
- Provides alternative position suggestions
- Position-specific rating calculations
- Detailed attribute scoring for 8 different positions

### 🌟 Professional Player Comparison
- Compares player stats with real professional footballers
- Shows similarity percentage with matching players
- Position-based weighted comparison
- Detailed stat comparison visualization

### 💰 Additional Features
- Transfer value estimation
- Unique player badges based on strongest attributes
- Interactive stat visualization with radar charts
- Responsive and modern UI design

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Django 5.2+
- TensorFlow 2.x
- Other dependencies (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fifaproject
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Apply database migrations:
```bash
python manage.py migrate
```

5. Run the development server:
```bash
python manage.py runserver
```

The application will be available at `http://localhost:8000`

## 🎯 Usage

1. Enter player attributes:
   - Physical (0-100)
   - Speed & Agility (0-100)
   - Attack (0-100)
   - Pass & Control (0-100)
   - Defense (0-100)
   - Free Kick (0-100)
   - Mental (0-100)

2. Upload a player photo (optional)

3. Choose a card template:
   - Gold Rare
   - TOTY (Team of the Year)
   - Silver
   - Bronze

4. Click "Prediksi" to generate:
   - FIFA-style player card
   - Optimal position recommendation
   - Alternative positions
   - Similar professional players
   - Estimated transfer value
   - Player badge

## 🛠️ Technical Details

### Position Scoring System
The application uses a sophisticated scoring system that considers:
- Position-specific attribute weights
- Minimum requirement thresholds
- Performance penalties
- Alternative position calculations

### Player Comparison Algorithm
Implements an advanced comparison system with:
- Position-based attribute weighting
- Normalized distance calculations
- Similarity scoring
- Position compatibility matching

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- FIFA for inspiration on the card design
- Real player data sourced from official football statistics
- Machine learning model trained on extensive football player dataset 