# Private Equity Data Analysis & Risk Assessment

## Overview
Comprehensive data analysis project for Private Equity (PE) portfolio management, including data transformation, risk analysis, and AI-enhanced reporting.

## Project Components

### 📊 Assignment 1: Data Transformation
Transform PE cash flow data from Excel format into normalized CSV structures for analysis.

**Key Features:**
- Multi-sheet Excel processing
- Automated metadata extraction
- Transaction normalization
- Vintage year calculation based on first capital call

**Output Files:**
- `metadata.csv` - Deal-level information (one row per deal)
- `measures.csv` - Transaction-level data (multiple rows per deal)

### 📈 Assignment 2: Risk Exposure Analysis
Analyze PE portfolio risk exposure relative to public markets and interest rates.

**Analysis Includes:**
- S&P 500 correlation analysis
- 10-year Treasury rate impact assessment
- OLS regression modeling
- Performance metrics (Beta, R-squared, Sharpe ratio)

**Data Sources:**
- Yahoo Finance API for market data
- Quarterly PE returns (2000-2008)

### 🤖 Bonus: AI-Enhanced Descriptions
Generate intelligent deal descriptions using LLM integration or algorithmic fallback.

**Features:**
- OpenAI API integration (optional)
- Algorithmic description generation
- Performance metrics calculation (DPI, TVPI)
- Deployment status assessment

## Installation

### Requirements
```bash
pip install pandas numpy yfinance matplotlib seaborn statsmodels openai
```

### Google Colab
All notebooks are optimized for Google Colab with built-in file upload functionality.

## Usage

### 1. Data Transformation
```python
# Run in Google Colab or locally
python transform.py "Private Equity Cash Flows.xlsx"
```

### 2. Risk Analysis
Upload PE returns data and run the analysis notebook to:
- Download market data automatically
- Generate correlation matrices
- Produce regression results
- Create visualization suite

### 3. AI Enhancement
```python
# With API key
export OPENAI_API_KEY="your-key"
python transform_with_ai.py

# Without API (uses algorithmic fallback)
python transform_with_ai.py
```

## Data Structure

### Metadata Schema
| Column | Description |
|--------|-------------|
| Deal Name | Unique identifier for PE investment |
| Vintage | Year of first capital call |
| Geography | Investment region |
| Asset Class | Investment type (P/E, Real Estate, etc.) |
| Commitment | Total committed capital |
| Capital Calls | Actual capital deployed |

### Measures Schema
| Column | Description |
|--------|-------------|
| Deal Name | Reference to metadata |
| Date | Transaction date |
| Investor | Investor identifier |
| Measure | Transaction type (Capital Call, Distribution, NAV) |
| Amount | Transaction value |

## Key Metrics

### Performance Indicators
- **DPI (Distributed to Paid-In)**: Distributions / Capital Calls
- **TVPI (Total Value to Paid-In)**: (Distributions + NAV) / Capital Calls
- **Deployment Rate**: Capital Calls / Commitment

### Risk Metrics
- **Beta**: Sensitivity to S&P 500 movements
- **R-squared**: Variance explained by market factors
- **Sharpe Ratio**: Risk-adjusted returns

## Results

### Risk Analysis Findings
- Correlation with public markets
- Interest rate sensitivity
- Statistical significance of market factors

### Portfolio Insights
- Deployment patterns across deals
- Geographic and asset class distribution
- Performance benchmarking

## File Structure
```
├── assignment1_data_transformation.ipynb
├── assignment2_risk_analysis.ipynb
├── bonus_assignment_llm.ipynb
├── transform.py
├── metadata.csv
├── measures.csv
└── pe_risk_analysis_data.csv
```

## Technologies Used
- **Python 3.8+**
- **pandas** - Data manipulation
- **yfinance** - Market data retrieval
- **statsmodels** - Statistical analysis
- **matplotlib/seaborn** - Visualization
- **OpenAI API** - AI descriptions (optional)

## Contributing
This project was developed as part of a PE data analysis initiative. For questions or improvements, please open an issue.

## License
MIT License - See LICENSE file for details

## Author
Data Analysis Professional specializing in Private Equity analytics

---
*Note: This project handles sensitive financial data. Ensure compliance with your organization's data policies.*
