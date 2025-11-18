#!/usr/bin/env python3
"""
Assignment 2: PE Risk Exposure Analysis
Analyze Private Equity portfolio risk relative to S&P 500 and interest rates
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from datetime import datetime
import warnings
import sys

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


def load_pe_data(file_path):
    """Load and prepare PE returns data"""
    pe_data = pd.read_excel(file_path)
    
    print("PE Data Overview")
    print("-"*50)
    print(f"Shape: {pe_data.shape}")
    print(f"Date range: {pe_data['end_date'].min()} to {pe_data['end_date'].max()}")
    
    # Calculate annualized return
    quarterly_mean = pe_data['PE - Buyout Appreciation'].mean()
    annualized = (1 + quarterly_mean)**4 - 1
    print(f"Mean quarterly return: {quarterly_mean:.2%}")
    print(f"Annualized return: {annualized:.2%}")
    
    # Prepare data
    pe_data['end_date'] = pd.to_datetime(pe_data['end_date'])
    pe_data = pe_data.rename(columns={
        'end_date': 'Date',
        'PE - Buyout Appreciation': 'PE_Return'
    })
    pe_data['Date'] = pe_data['Date'] + pd.offsets.QuarterEnd(0)
    pe_data = pe_data.set_index('Date')
    
    return pe_data


def fetch_market_data(start_date='1999-12-01', end_date='2009-03-31'):
    """Fetch S&P 500 and Treasury data from Yahoo Finance"""
    print("\nFetching market data...")
    
    # Download S&P 500
    print("Downloading S&P 500...")
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    
    sp500_quarterly = sp500['Close'].resample('Q').last()
    sp500_returns = pd.DataFrame({
        'SP500_Price': sp500_quarterly,
        'SP500_Return': sp500_quarterly.pct_change()
    })
    print(f"  S&P 500: {len(sp500_returns)} quarters")
    
    # Download Treasury rates
    print("Downloading Treasury rates...")
    tnx = yf.download('^TNX', start=start_date, end=end_date, progress=False)
    
    if isinstance(tnx.columns, pd.MultiIndex):
        tnx.columns = tnx.columns.get_level_values(0)
    
    tnx_quarterly = tnx['Close'].resample('Q').mean()
    treasury_data = pd.DataFrame({
        'Treasury_Rate': tnx_quarterly / 100,
        'Rate_Change': (tnx_quarterly / 100).diff()
    })
    print(f"  Treasury: {len(treasury_data)} quarters")
    
    return sp500_returns, treasury_data


def perform_correlation_analysis(merged_data):
    """Calculate correlation matrix and key metrics"""
    corr = merged_data[['PE_Return', 'SP500_Return', 'Treasury_Rate']].corr()
    
    print("\nCorrelation Matrix:")
    print(corr.round(3))
    
    pe_sp500_corr = corr.loc['PE_Return', 'SP500_Return']
    pe_rate_corr = corr.loc['PE_Return', 'Treasury_Rate']
    
    print(f"\nKey Correlations:")
    print(f"  PE vs S&P 500: {pe_sp500_corr:.3f}")
    print(f"  PE vs Treasury: {pe_rate_corr:.3f}")
    
    return corr, pe_sp500_corr, pe_rate_corr


def perform_regression_analysis(merged_data):
    """Run OLS regression analysis"""
    X = merged_data[['SP500_Return', 'Treasury_Rate', 'Rate_Change']]
    y = merged_data['PE_Return']
    
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    
    print("\nRegression Results Summary:")
    print("-"*50)
    print(f"R-squared: {model.rsquared:.3f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.3f}")
    print(f"F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.4f})")
    
    print("\nCoefficients:")
    for var, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
        significance = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {var:15} {coef:8.4f} (p={pval:.4f}) {significance}")
    
    return model


def create_visualizations(merged_data, corr, pe_sp500_corr):
    """Create comprehensive visualization suite"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Time series
    ax1 = axes[0, 0]
    ax1.plot(merged_data.index, merged_data['PE_Return']*100, label='PE', linewidth=2, color='#2E86AB')
    ax1.plot(merged_data.index, merged_data['SP500_Return']*100, label='S&P 500', alpha=0.7, color='#A23B72')
    ax1.set_title('Returns Over Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Return (%)')
    ax1.set_xlabel('Date')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. PE vs S&P scatter
    ax2 = axes[0, 1]
    ax2.scatter(merged_data['SP500_Return']*100, merged_data['PE_Return']*100, alpha=0.6, color='#2E86AB')
    
    # Add regression line
    z = np.polyfit(merged_data['SP500_Return'], merged_data['PE_Return'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_data['SP500_Return'].min(), merged_data['SP500_Return'].max(), 100)
    ax2.plot(x_line*100, p(x_line)*100, 'r--', label=f'β={z[0]:.2f}', linewidth=2)
    
    ax2.set_xlabel('S&P 500 Return (%)')
    ax2.set_ylabel('PE Return (%)')
    ax2.set_title('PE vs S&P 500 Returns', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. PE vs Interest Rate
    ax3 = axes[1, 0]
    ax3.scatter(merged_data['Treasury_Rate']*100, merged_data['PE_Return']*100, alpha=0.6, color='#F18F01')
    ax3.set_xlabel('10-Year Treasury Rate (%)')
    ax3.set_ylabel('PE Return (%)')
    ax3.set_title('PE vs Interest Rates', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation heatmap
    ax4 = axes[1, 1]
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                ax=ax4, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": .8})
    ax4.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    plt.suptitle('PE Risk Exposure Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def generate_executive_summary(merged_data, model, pe_sp500_corr, pe_rate_corr):
    """Generate executive summary of findings"""
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY: PE RISK EXPOSURE ANALYSIS")
    print("="*60)
    
    pe_mean = merged_data['PE_Return'].mean()
    pe_std = merged_data['PE_Return'].std()
    sp500_mean = merged_data['SP500_Return'].mean()
    sp500_std = merged_data['SP500_Return'].std()
    
    print("\nPerformance Metrics:")
    print(f"  PE Quarterly Return: {pe_mean:.2%} (σ={pe_std:.2%})")
    print(f"  PE Annualized Return: {(1+pe_mean)**4-1:.2%}")
    print(f"  PE Sharpe Ratio: {pe_mean/pe_std:.2f}")
    print(f"  S&P 500 Quarterly Return: {sp500_mean:.2%} (σ={sp500_std:.2%})")
    
    print("\nKey Findings:")
    print(f"  1. Market Correlation: {pe_sp500_corr:.3f}")
    print(f"  2. R-squared: {model.rsquared:.3f} ({model.rsquared*100:.1f}% variance explained)")
    
    sp500_coef = model.params['SP500_Return']
    sp500_pval = model.pvalues['SP500_Return']
    print(f"  3. Market Beta: {sp500_coef:.2f} (p={sp500_pval:.3f})")
    
    if sp500_pval < 0.05:
        print(f"     → Statistically significant at 5% level")
        print(f"     → 1% S&P change → {sp500_coef:.2f}% PE change")
    
    print("\nRisk Assessment:")
    if abs(pe_sp500_corr) > 0.5:
        print("  • High market correlation - significant systematic risk")
    elif abs(pe_sp500_corr) > 0.3:
        print("  • Moderate market correlation - balanced risk profile")
    else:
        print("  • Low market correlation - good diversification")
    
    if pe_rate_corr < -0.3:
        print("  • Negative rate sensitivity - vulnerable to rising rates")
    
    print("\nRecommendations:")
    if abs(pe_sp500_corr) > 0.5:
        print("  • Consider hedging strategies for market downturns")
    print("  • Focus on operational value creation")
    print("  • Implement dynamic risk management")
    print("  • Monitor Federal Reserve policy closely")
    
    print("\n" + "="*60)


def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python pe_risk_analysis.py <pe_data.xlsx>")
        print("\nThis script analyzes PE portfolio risk exposure")
        sys.exit(1)
    
    pe_file = sys.argv[1]
    
    print("PE RISK EXPOSURE ANALYSIS")
    print("="*60)
    
    # Load PE data
    pe_data = load_pe_data(pe_file)
    
    # Fetch market data
    sp500_returns, treasury_data = fetch_market_data()
    
    # Merge datasets
    merged = pd.concat([pe_data, sp500_returns, treasury_data], axis=1, join='inner')
    merged = merged.dropna()
    print(f"\nMerged dataset: {len(merged)} quarters")
    
    # Perform analyses
    corr, pe_sp500_corr, pe_rate_corr = perform_correlation_analysis(merged)
    model = perform_regression_analysis(merged)
    
    # Create visualizations
    fig = create_visualizations(merged, corr, pe_sp500_corr)
    
    # Generate summary
    generate_executive_summary(merged, model, pe_sp500_corr, pe_rate_corr)
    
    # Save outputs
    merged.to_csv('pe_risk_analysis_data.csv')
    fig.savefig('pe_risk_analysis.png', dpi=300, bbox_inches='tight')
    
    # Save regression results
    with open('regression_results.txt', 'w') as f:
        f.write(str(model.summary()))
    
    print("\nOutput files created:")
    print("  - pe_risk_analysis_data.csv")
    print("  - pe_risk_analysis.png")
    print("  - regression_results.txt")
    
    plt.show()


if __name__ == "__main__":
    main()
