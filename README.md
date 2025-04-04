# CreditWise

<div align="center">
  <img src="https://github.com/seu-usuario/creditwise/raw/main/assets/logo.png" alt="CreditWise Logo" width="400px">
  
  **Advanced Credit Risk Analysis with ML Explainability**
  
  [![Python](https://img.shields.io/badge/Python-3.7%2B-4584b6?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![SHAP](https://img.shields.io/badge/SHAP-0.41.0-00C244?style=flat-square)](https://github.com/slundberg/shap)
  [![XGBoost](https://img.shields.io/badge/XGBoost-1.7.5-0073B7?style=flat-square)](https://xgboost.readthedocs.io/)
  [![LightGBM](https://img.shields.io/badge/LightGBM-3.3.5-3498DB?style=flat-square)](https://lightgbm.readthedocs.io/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
  [![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000?style=flat-square)](https://github.com/psf/black)
</div>

## Overview

CreditWise is a credit risk analysis platform that uses machine learning and explainability techniques to deliver credit assessments and actionable insights. The application leverages synthetic data to demonstrate credit scoring and collection strategies.

### Core Capabilities

1. **Data Generation & Processing** - Creates synthetic customer profiles with financial attributes and payment histories 
2. **ML Model Deployment** - Implements ensemble models (Random Forest, XGBoost, LightGBM, MLP, LogisticRegression) for credit default prediction
3. **Explainable AI** - Utilizes SHAP to provide transparent model decisions with global and local explanations
4. **Portfolio Simulation** - Tests approval strategies and economic scenarios to optimize risk-return metrics
5. **Client Analysis** - Delivers individual risk assessment with factor breakdown
6. **Collection Simulation** - Optimizes individual debt collection strategies through risk-based approaches

## Current Project Structure

```
creditwise/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # English documentation
├── README.pt-BR.md        # Portuguese documentation
└── creditwise_env/        # Python virtual environment
```

## Implementation Details

### Data Management

- Synthetic customer profiles with 10+ financial attributes (age, income, employment time, etc.)
- Generation of synthetic collection data for defaulted customers
- Preprocessing including scaling and train-test splitting

### Analytics Engine

- Multiple model implementations (Random Forest, XGBoost, LightGBM, MLP, LogisticRegression)
- Performance metrics: ROC-AUC, precision-recall, F1 score
- Cross-validation for model stability assessment
- Feature importance analysis and ranking

### Interactive Interface

- Portfolio risk distribution visualization
- Model performance dashboards with detailed metrics
- Exploratory data analysis tools
- Correlation analysis and variable distributions

### Decision Explainability

- Global feature importance visualization
- SHAP dependency plots for feature relationship analysis
- Individual prediction explanations for transparent decisions
- Technical insights translated into business-actionable recommendations

### Risk Analysis

- Approval threshold testing with business metric impact analysis
- Portfolio performance analysis
- Risk-return trade-off visualization
- Model comparison with performance metrics

### Collection Strategies

- Collection performance analysis
- Channel effectiveness comparison
- Recovery time analysis
- Cost-benefit calculations for collection efforts

## Application Modules

### Overview
- Credit scoring introduction
- Score distribution visualization
- Key performance metrics

### Data Exploration
- Descriptive statistics
- Variable distributions
- Correlation analysis
- Bivariate analysis

### Credit Model
- Model performance metrics
- ROC curve analysis
- Feature importance visualization
- Decision threshold analysis

### Model Comparison
- Performance metrics across multiple algorithms
- Cross-validation results
- Confusion matrix comparison

### Explainability (SHAP)
- Global feature importance
- Feature dependency plots
- Local decision explanations
- Individual client analysis

### Credit Simulator
- Client profile creation
- Credit score prediction
- Decision factors analysis
- What-if scenario testing

### Collection Analysis
- Defaulted client profiles
- Collection strategy evaluation
- Recovery simulation
- ROI analysis for collection efforts

## Performance Benchmarks

| Model | ROC-AUC | Precision | Recall | F1 Score |
|-------|---------|-----------|--------|----------|
| Random Forest | 0.82 | 0.76 | 0.71 | 0.73 |
| XGBoost | 0.85 | 0.79 | 0.73 | 0.76 |
| LightGBM | 0.84 | 0.78 | 0.72 | 0.75 |
| MLP | 0.81 | 0.74 | 0.70 | 0.72 |
| Logistic Regression | 0.79 | 0.73 | 0.69 | 0.71 |

### Primary Risk Factors

The models identify these key predictors for credit default risk:
1. Payment history (delinquency patterns)
2. Credit utilization rate
3. Debt-to-income ratio
4. Employment stability
5. Credit inquiry frequency

## Technology Stack

- **Core**: Python
- **Data Processing**: Pandas, NumPy
- **ML Framework**: Scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn

## Installation

### Requirements

- Python 3.7+
- pip package manager
- Virtual environment (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/creditwise.git
cd creditwise

# Create and activate virtual environment
python -m venv creditwise_env
source creditwise_env/bin/activate  # On Windows: creditwise_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

The application will be accessible at http://localhost:8501

## Contributing

We welcome contributions that enhance CreditWise's capabilities:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## License

Released under MIT License. See `LICENSE` for details.

## Contact

Project Link: [https://github.com/your-username/creditwise](https://github.com/your-username/creditwise)

---

<div align="center">
  <p>
    <b>CreditWise</b> - Credit Risk Analysis with Explainable AI
  </p>
</div> 
