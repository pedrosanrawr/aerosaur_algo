# AEROSAUR Smart Mode Machine Learning

AI-powered air purifier system that uses Machine Learning to automatically adjust fan speed based on environmental conditions.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Technical Specifications](#technical-specifications)

---

## Overview

This project implements a Random Forest Classifier to predict optimal fan speed settings for the AEROSAUR air purification system. The model analyzes real-time air quality data and automatically adjusts fan speed to maintain optimal air quality while minimizing energy consumption.

### Key Features

- Real-time air quality monitoring and prediction
- Automated fan speed adjustment (Off, Low, Medium, High)
- High accuracy classification (98.5%)
- Scalable architecture supporting 500,000+ records
- IoT integration ready (ESP32 compatible)

---

## Prerequisites

### System Requirements

- **Python**: Version 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **RAM**: Minimum 4GB (8GB recommended for large datasets)
- **Storage**: Minimum 100MB free space

### Required Software

- Python 3.8+
- pip (Python package manager)
- Jupyter Notebook or JupyterLab

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/pedrosanrawr/aerosaur_algo.git
cd aerosaur_algo
```

### Step 2: Create a Virtual Environment (Recommended)

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

**Manual Installation (if requirements.txt is not available):**
```bash
pip install pandas numpy scikit-learn matplotlib joblib jupyter
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Your default browser will open automatically. Navigate to `aerosaur_algo.ipynb` and open it.

---

## Usage

### Running the Notebook

1. Open `aerosaur_algo.ipynb` in Jupyter Notebook
2. Run all cells sequentially from top to bottom
3. Use **Cell > Run All** or press **Shift + Enter** for each cell

### Training the Model

The notebook will automatically:
1. Generate synthetic training data (1,000 samples)
2. Train the Random Forest Classifier
3. Evaluate model performance
4. Display accuracy metrics and visualizations

### Making Predictions

After training, you can make predictions by modifying the prediction cells with your own input values:

```python
# Example prediction
aqi = 180
aqi_noisy = aqi + np.random.normal(0, 2)
aqi_prev = 175
isPurifierOn = 1

sample = pd.DataFrame(
    [[21, aqi_noisy, aqi_prev, isPurifierOn]],
    columns=['time', 'aqi_noisy', 'aqi_prev', 'isPurifierOn']
)

prediction = model.predict(sample)
print("Predicted Fan Speed:", prediction[0])
```

---

## Project Structure

```
aerosaur_algo/
│
├── aerosaur_algo.ipynb          # Main Jupyter notebook
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore file
```

---

## Model Details

### Algorithm

**Random Forest Classifier**

A supervised machine learning algorithm that uses ensemble learning with multiple decision trees to improve prediction accuracy and prevent overfitting.

### Input Features

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| `time` | Hour of the day | Integer | 0-23 |
| `aqi_noisy` | Air Quality Index with sensor noise | Float | 0-300+ |
| `aqi_prev` | Previous AQI reading | Float | 0-300+ |
| `isPurifierOn` | Purifier operational status | Binary | 0 or 1 |

### Output Classes

| Class | Fan Speed | Description |
|-------|-----------|-------------|
| 0 | Off | Purifier or fan disabled |
| 1 | Low | Minimal air circulation |
| 2 | Medium | Moderate air filtration |
| 3 | High | Maximum air purification |

### Model Configuration

- **Number of Trees**: 100
- **Train/Test Split**: 80% / 20%
- **Random State**: 42 (for reproducibility)
- **Model Accuracy**: 98.50%

---

## Technical Specifications

### Time Complexity

**Training:**
```
O(n × m × log(n) × k)

Where:
- n = number of training samples
- m = number of features (4)
- k = number of trees (100)
- log(n) = average tree depth
```

**Prediction:**
```
O(k × log(n))

Approximately 1-5 milliseconds per prediction
```

### Space Complexity

- **Training Data**: O(n × m)
- **Model Size**: 1-2 MB
- **Memory Usage**: Minimal, suitable for edge deployment

### Performance Benchmarks

| Dataset Size | Training Time | Prediction Time (100K samples) | Accuracy |
|--------------|---------------|--------------------------------|----------|
| 1,000 | ~3 seconds | ~0.01 seconds | 98.50% |
| 500,000 | ~8 seconds | ~0.11 seconds | 98.73% |

---

## Troubleshooting

### Issue: "Command not found: jupyter"

**Solution:**
```bash
python3 -m notebook
```

### Issue: Permission errors during installation

**Solution:**
```bash
pip install --user pandas numpy scikit-learn matplotlib joblib jupyter
```

### Issue: Kernel crashes with large datasets

**Solution:**
- Reduce dataset size in test cells
- Increase available RAM
- Use `n_jobs=-1` for parallel processing in RandomForestClassifier

### Issue: Module import errors

**Solution:**
Ensure all dependencies are installed:
```bash
pip install --upgrade pandas numpy scikit-learn matplotlib joblib
```

---

## Dependencies

The project requires the following Python packages:

- **pandas** (>= 1.3.0): Data manipulation and analysis
- **numpy** (>= 1.21.0): Numerical computing
- **scikit-learn** (>= 1.0.0): Machine learning algorithms
- **matplotlib** (>= 3.4.0): Data visualization
- **joblib** (>= 1.0.0): Model serialization
- **jupyter** (>= 1.0.0): Interactive notebook environment

---

## Integration with IoT Systems

### ESP32 Deployment

The trained model can be integrated with ESP32 microcontrollers:

1. Train model using this notebook
2. Save model using `joblib.dump(model, 'model.joblib')`
3. Deploy model to cloud server (Flask/FastAPI)
4. ESP32 sends sensor data via WiFi to server
5. Server returns predicted fan speed
6. ESP32 adjusts PWM signal to motor

### Example Integration Architecture

```
[ESP32 + Sensors] --WiFi--> [Cloud Server + ML Model] --WiFi--> [ESP32 Motor Control]
```

