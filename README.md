# CreditWise

<div align="center">
  <img src="https://github.com/seu-usuario/creditwise/raw/main/assets/logo.png" alt="CreditWise Logo" width="400px">
  
  **Advanced Credit Risk Analysis with ML Explainability**
  
  [![Python](https://img.shields.io/badge/Python-3.7%2B-4584b6?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![SHAP](https://img.shields.io/badge/SHAP-Latest-00C244?style=flat-square)](https://github.com/slundberg/shap)
  [![XGBoost](https://img.shields.io/badge/XGBoost-Latest-0073B7?style=flat-square)](https://xgboost.readthedocs.io/)
  [![LightGBM](https://img.shields.io/badge/LightGBM-Latest-3498DB?style=flat-square)](https://lightgbm.readthedocs.io/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
  [![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000?style=flat-square)](https://github.com/psf/black)
</div>

## Overview

CreditWise is an enterprise-grade credit risk analysis platform that leverages machine learning and advanced explainability techniques to deliver precise credit assessments and actionable insights.

### Core Capabilities

1. **Data Generation & Processing** - Creates synthetic customer profiles with financial attributes and payment histories using advanced preprocessing pipelines
2. **ML Model Deployment** - Implements ensemble models (Random Forest, XGBoost, LightGBM) for accurate default prediction
3. **Explainable AI** - Utilizes SHAP to provide transparent model decisions with global and local explanations
4. **Portfolio Simulation** - Tests approval strategies and economic scenarios to optimize risk-return metrics
5. **Client Analysis** - Delivers in-depth individual risk assessment with factor breakdown
6. **Collection Simulation** - Optimizes individual debt collection strategies through risk-based approaches
7. **Cash Flow Forecasting** - Projects future portfolio performance under various economic conditions
8. **Economic Scenario Analysis** - Simulates the impact of different economic scenarios on portfolio health

## Technical Architecture

```
creditwise/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── assets/                 # Static assets
├── models/                 # Pre-trained models
│   └── creditwise_model.pkl
├── utils/                  # Utility functions
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── model_utils.py
│   └── visualization.py
└── notebooks/              # Development notebooks
    ├── data_exploration.ipynb
    └── model_development.ipynb
```

## Implementation Details

### Data Management

- Synthetic customer profiles with 20+ financial attributes
- Comprehensive credit history simulation with behavioral patterns
- Advanced preprocessing including scaling, encoding and feature engineering
- Class imbalance handling for robust model training
- Synthetic collection data generation for recovery simulation

### Analytics Engine

- Ensemble models with automated hyperparameter optimization
- Multiple model implementations (Random Forest, XGBoost, LightGBM, MLP, LogisticRegression)
- Performance metrics: ROC-AUC, precision-recall, F1 score
- Cross-validation for model stability assessment
- Feature importance analysis and ranking

### Interactive Interface

- Portfolio risk distribution visualization
- Model performance dashboards with detailed metrics
- Segment-specific analysis with filtering capabilities
- Historical trend analysis and comparisons
- Dynamic credit policy simulation

### Decision Explainability

- Global feature importance visualization
- SHAP dependency plots for feature relationship analysis
- Individual prediction explanations for transparent decisions
- Technical insights translated into business-actionable recommendations
- Customer segment comparison with population benchmarks

### Risk Optimization

- Approval threshold testing with business metric impact analysis
- Multiple economic scenario simulation (baseline, optimistic, pessimistic, crisis)
- Portfolio performance projections under varying conditions
- Risk-return trade-off visualization
- Strategy comparison with actionable recommendations

### Individual Assessment

- Detailed client profile examination
- Risk factor identification and quantification
- Population average comparisons
- What-if scenario analysis for individual clients
- Personalized approval limit recommendations

### Collection Strategy Simulator

- **Personalized Collection Approach** - Tailors collection strategies based on individual risk profiles and payment behavior
- **ROI Optimization** - Calculates expected return on collection efforts based on client characteristics and debt amount
- **Contact Priority Scheduling** - Generates optimal contact schedules (timing/frequency) based on client responsiveness patterns
- **Payment Plan Generation** - Creates customized payment plans considering client capacity and payment probability
- **Simulation Comparison** - Tests different collection approaches side-by-side to identify optimal strategy
- **Success Probability Scoring** - Provides probability metrics for successful collection through different channels
- **Cost-Benefit Analysis** - Performs detailed cost analysis of collection actions against expected recovery values

### Economic Impact Analysis

- **Macroeconomic Scenario Planning** - Simulates portfolio performance under different economic conditions
- **Stress Testing** - Tests portfolio resilience under extreme market conditions
- **Interest Rate Sensitivity** - Analyzes the impact of interest rate changes on portfolio profitability
- **Unemployment Impact Modeling** - Projects default rates based on changing unemployment conditions
- **GDP Growth Correlation** - Models the relationship between economic growth and credit performance
- **Risk Segment Response** - Analyzes how different risk segments react to economic changes
- **Recommendation Engine** - Provides strategic portfolio adjustments based on economic forecasts

## Advanced Dashboards & Specialized Features

### Credit Assessment Dashboards

1. **Multi-Level Scoring Dashboard**
   - **Score Distribution Map** - Geographic visualization of credit scores across regions
   - **Vintage Analysis** - Performance tracking of credit accounts by origination period
   - **Revolving vs. Installment Dashboard** - Comparative analysis of different credit products
   - **Credit Line Utilization Tracker** - Real-time monitoring of credit usage patterns

2. **Underwriting Decision Support**
   - **Auto-Decisioning Panel** - Rules-based approval visualization with override tracking
   - **Document Verification Status** - Tracks customer documentation completeness
   - **Policy Exception Dashboard** - Monitors and analyzes exceptions to standard credit policies
   - **Multi-Bureau Comparison** - Side-by-side analysis of data from multiple credit bureaus

3. **Pricing Optimization Tools**
   - **Risk-Based Pricing Simulator** - Tests different pricing strategies across risk tiers
   - **Elasticity Analysis** - Measures customer sensitivity to rate changes
   - **Competitive Rate Analysis** - Benchmarks offerings against market alternatives
   - **Fee Structure Optimization** - Simulates revenue impact of fee adjustments

### Collection Intelligence Center

1. **Collection Performance Dashboards**
   - **Recovery Funnel Visualization** - Tracks accounts through various collection stages
   - **Agent Performance Metrics** - Detailed KPIs for collection team management
   - **Time-to-Recover Analysis** - Measures efficiency of collection strategies over time
   - **Channel Effectiveness Comparison** - Analyzes performance across contact methods

2. **Behavior-Based Collection Tools**
   - **Payment Promise Tracker** - Monitors fulfillment of customer payment commitments
   - **Communication Response Analyzer** - Measures customer engagement with collection attempts
   - **Optimal Contact Time Predictor** - Identifies ideal timing for customer outreach
   - **Digital Engagement Dashboard** - Tracks customer interaction with digital collection channels

3. **Settlement Strategy Workbench**
   - **Discount Authorization Matrix** - Framework for approval of settlement offers
   - **NPV Settlement Calculator** - Evaluates long-term value of settlement options
   - **Legal Action Assessment** - Cost-benefit analysis of legal proceedings
   - **Restructuring Impact Simulator** - Projects outcomes of loan modification scenarios

### Portfolio Management Command Center

1. **Executive Analytics Suite**
   - **Consolidated Risk Metrics** - Holistic view of portfolio health indicators
   - **Vintage Curve Analyzer** - Tracks performance evolution across origination periods
   - **Concentration Risk Heatmap** - Identifies over-exposure to specific segments
   - **Profitability Attribution Model** - Breaks down profit drivers across the portfolio

2. **Early Warning System**
   - **Behavioral Red Flag Detector** - Identifies concerning pattern changes before default
   - **Macroeconomic Impact Simulator** - Projects portfolio performance under changing conditions
   - **Cross-Default Contagion Map** - Visualizes correlations between defaults across segments
   - **Forward-Looking Indicators** - Predictive metrics for portfolio quality trends

3. **Strategic Planning Tools**
   - **Portfolio Restructuring Simulator** - Tests impact of major policy shifts
   - **Capital Allocation Optimizer** - Recommends optimal distribution of lending resources
   - **Growth Scenario Planner** - Models various portfolio expansion strategies
   - **Stress Testing Workbench** - Comprehensive tools for regulatory and internal stress tests

4. **Cash Flow Projection Tools**
   - **Portfolio Revenue Forecast** - Projects expected future revenue from the loan portfolio
   - **Default Rate Simulator** - Models expected defaults under various economic conditions
   - **Recovery Rate Calculator** - Estimates recovery expectations for defaulted debt
   - **Net Cash Flow Dashboard** - Visualizes expected inflows and outflows over time periods
   - **Liquidity Planning Tool** - Helps optimize timing of funding needs based on portfolio projections

## Advanced API Integrations

- **Open Banking Connectors** - Seamless integration with financial account aggregation services
- **Payment Gateway Integration** - Direct connections to facilitate immediate payments
- **Credit Bureau API** - Real-time access to credit reporting data
- **Document OCR Processing** - Automated extraction of information from financial documents
- **Customer Communication Platform** - Multi-channel outreach capabilities (email, SMS, voice)
- **Legal Service Integration** - Automated management of legal collection processes
- **Regulatory Compliance Checker** - Ensures all collection activities meet legal requirements

## Key Recovery & Pricing Tools in Detail

### Time-to-Recovery Analysis
The Time-to-Recovery Analysis module offers comprehensive temporal insights into the collection process:

- **Recovery Curves Visualization**: Interactive graphs showing recovery percentages over time periods
- **Segment-Based Recovery Timelines**: Detailed analysis of collection timeframes by:
  - Debt amount ranges
  - Customer risk profiles
  - Product categories
  - Geographic regions
- **Acceleration Point Detection**: AI-powered identification of critical moments where recovery can be expedited
- **Recovery Forecasting Engine**: Machine learning models that project when specific portfolio percentages will be recovered
- **Statute of Limitations Monitoring**: Automated alerting system for accounts approaching legal time limits
- **Recovery Velocity Metrics**: Comparative analysis of collection speed across different strategies

### Channel Effectiveness Comparison
This analytics suite measures and optimizes communication channel performance:

- **Multi-Channel Response Analytics**: Side-by-side comparison of customer engagement across:
  - SMS campaigns
  - Email sequences
  - Voice calls
  - Digital messaging (WhatsApp, Telegram)
  - Physical mail
  - In-person visits
- **Cost-Per-Contact Calculation**: Detailed breakdown of resource investment by channel
- **Channel ROI Dashboard**: Visual representation of return on investment for each communication method
- **Customer Preference Mapping**: Machine learning models that identify optimal channels by customer segment
- **Sequential Strategy Builder**: AI-assisted tool for designing multi-channel contact sequences
- **Temporal Effectiveness Heatmap**: Visualization of ideal contact times for each channel and customer segment

### Discount Authorization Matrix
This enterprise governance tool establishes a structured framework for discount approvals:

- **Hierarchical Approval Levels**: Configurable authorization tiers based on:
  - Discount percentage thresholds
  - Absolute amount thresholds
  - Portfolio segment considerations
- **Parameter-Driven Rules Engine**: Automated application of discount policies based on:
  - Delinquency duration
  - Total outstanding amount
  - Customer relationship history
  - Recovery probability scoring
- **Approval Workflow Automation**: End-to-end process management for exception handling
- **Exception Audit Trail**: Comprehensive logging system for compliance and performance analysis
- **Discount Impact Monitoring**: Real-time dashboards showing financial implications of applied discounts
- **Policy Simulation Environment**: Sandbox for testing matrix adjustments before production deployment

### NPV Settlement Calculator
This financial analysis tool evaluates the true value of settlement options:

- **Time-Value Adjustment Engine**: Sophisticated NPV calculations incorporating:
  - Customizable discount rates
  - Risk-adjusted cash flow projections
  - Seasonal payment pattern modeling
- **Settlement Scenario Comparison**: Side-by-side evaluation of different arrangements:
  - Lump sum with discount
  - Entry payment plus installments
  - Extended payment plans
  - Interest/fee forgiveness options
- **Compliance Parameter Integration**: Automatic consideration of regulatory constraints
- **Settlement Probability Modeling**: ML-based prediction of fulfillment likelihood for each option
- **Optimal Recommendation Engine**: AI-powered suggestion of most valuable settlement structure
- **Negotiation Support Interface**: Real-time calculator for use during customer interactions

### Legal Action Cost-Benefit Analysis
This decision support system evaluates the economic viability of legal proceedings:

- **Comprehensive Cost Modeling**: Detailed calculation of all legal expenses:
  - Court filing fees by jurisdiction
  - Attorney fee structures (fixed, hourly, contingency)
  - Expert witness costs
  - Administrative expenses
  - Opportunity cost of capital
- **Success Probability Assessment**: ML-based prediction of outcomes based on:
  - Case type and jurisdiction
  - Historical precedents
  - Debtor profile and assets
  - Documentation quality
- **Expected Value Calculator**: Sophisticated analysis of risk-adjusted returns
- **Alternative Strategy Comparison**: Evaluation of legal action against other recovery options
- **Case Prioritization Algorithm**: Automated ranking of cases by expected ROI
- **Minimum Threshold Recommender**: AI-driven recommendations for minimum viable case value

### Debt Restructuring Impact Simulator
This advanced modeling tool projects outcomes of loan modification scenarios:

- **Restructuring Scenario Modeling**: Comprehensive simulation of options:
  - Term extensions with payment recalculation
  - Interest rate modifications
  - Debt consolidation scenarios
  - Principal forgiveness impacts
  - Payment holiday effects
- **Cash Flow Impact Visualization**: Time-series projections of payment streams
- **Affordability Analysis**: Assessment of customer payment capacity using:
  - Income and expense data
  - Debt service ratio calculation
  - Payment history patterns
  - Behavioral scoring
- **Default Probability Recalculation**: ML models that predict completion likelihood
- **Accounting Impact Assessment**: Calculation of book value changes and P&L effects
- **Contract Document Generator**: Automated creation of restructuring agreements with terms

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
2. Debt-to-income ratio
3. Credit utilization rate
4. Employment stability
5. Credit inquiry frequency

## Technology Stack

- **Core**: Python
- **Data Processing**: Pandas, NumPy
- **ML Framework**: Scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Development**: Jupyter

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
    <b>CreditWise</b> - Precision Credit Risk Analysis with Explainable AI
  </p>
</div> 
