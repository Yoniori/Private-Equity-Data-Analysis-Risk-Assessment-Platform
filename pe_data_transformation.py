#!/usr/bin/env python3
"""
Assignment 1: PE Data Transformation
Transform Private Equity cash flow data from Excel to normalized CSV format
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')


def load_excel_data(excel_file):
    """Load all sheets from Excel file"""
    excel_data = pd.read_excel(excel_file, sheet_name=None)
    print(f"Loaded {len(excel_data)} sheets from {excel_file}")
    
    for sheet_name, df in excel_data.items():
        print(f"  - {sheet_name}: {df.shape}")
    
    return excel_data


def create_metadata(excel_data):
    """Create metadata DataFrame with deal-level information"""
    metadata_list = []
    
    for sheet_name, df in excel_data.items():
        # Skip Summary sheet and sheets without Date column
        if sheet_name == 'Summary' or 'Date' not in df.columns:
            continue
        
        deal_name = sheet_name.strip()
        
        # Convert dates
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        
        # Calculate total commitment
        commitment_cols = [col for col in df.columns if 'Commitment' in col]
        total_commitment = sum(df[col].sum() for col in commitment_cols if col in df.columns)
        
        # Calculate total capital calls
        capital_call_cols = [col for col in df.columns if 'Capital Call' in col and 'Chargeback' not in col]
        total_capital_calls = sum(df[col].sum() for col in capital_call_cols if col in df.columns)
        
        # Find first capital call date (commitment date)
        first_call_date = None
        for col in capital_call_cols:
            if col in df.columns:
                valid_dates = df[df[col] > 0]['Date'].dropna()
                if not valid_dates.empty:
                    date = valid_dates.min()
                    if first_call_date is None or date < first_call_date:
                        first_call_date = date
        
        vintage = first_call_date.year if first_call_date else None
        
        # Try to get geography and asset class from Summary sheet
        geography = 'USA'
        asset_class = 'P/E'
        
        if 'Summary' in excel_data and 'Deal Name' in excel_data['Summary'].columns:
            summary_df = excel_data['Summary']
            deal_info = summary_df[summary_df['Deal Name'] == deal_name]
            
            if not deal_info.empty:
                if 'Geography' in deal_info.columns:
                    geography = deal_info['Geography'].iloc[0]
                if 'Asset Class' in deal_info.columns:
                    asset_class = deal_info['Asset Class'].iloc[0]
        
        # Create metadata record
        metadata_list.append({
            'Deal Name': deal_name,
            'Management Fees': 0,
            'Commitment Date': first_call_date.strftime('%m/%d/%Y') if first_call_date else '',
            'Vintage': vintage,
            'Currency': 'USD',
            'Geography': geography,
            'Asset Class': asset_class,
            'Underlying': '',
            'PE Tags': '',
            'Commitment': total_commitment,
            'Capital Calls': total_capital_calls
        })
    
    metadata_df = pd.DataFrame(metadata_list)
    print(f"\nCreated metadata for {len(metadata_df)} deals")
    
    return metadata_df


def create_measures(excel_data):
    """Create measures DataFrame with transaction-level data"""
    measures_list = []
    
    for sheet_name, df in excel_data.items():
        # Skip Summary sheet and sheets without Date column
        if sheet_name == 'Summary' or 'Date' not in df.columns:
            continue
        
        deal_name = sheet_name.strip()
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        
        # Process each column
        for col in df.columns:
            if col == 'Date' or not isinstance(col, str):
                continue
            
            measure_type = None
            investor = col
            
            # Identify measure type
            if 'Capital Call' in col:
                if 'Chargeback' in col:
                    measure_type = 'Chargeback Capital Call'
                    investor = col.replace(' Chargeback Capital Call', '').replace(' Chargeback', '')
                else:
                    measure_type = 'Capital Call'
                    investor = col.replace(' Capital Call', '')
            elif 'Distribution' in col:
                measure_type = 'Distribution'
                investor = col.replace(' Distribution', '')
            elif 'Commitment' in col:
                measure_type = 'Commitment'
                investor = col.replace(' Commitment', '')
            elif 'Estimated NAV' in col or 'NAV' in col:
                measure_type = 'Estimated NAV'
                investor = col.replace(' Estimated NAV', '').replace(' NAV', '')
            else:
                continue
            
            # Add non-zero measures
            for idx, row in df.iterrows():
                if pd.notna(row[col]) and row[col] != 0:
                    measures_list.append({
                        'Deal Name': deal_name,
                        'Date': row['Date'].strftime('%m/%d/%Y') if pd.notna(row['Date']) else '',
                        'Investor': investor.strip(),
                        'Measure': measure_type,
                        'Amount': float(row[col])
                    })
    
    measures_df = pd.DataFrame(measures_list)
    print(f"Created {len(measures_df)} measure records")
    
    if not measures_df.empty:
        print(f"Unique measure types: {measures_df['Measure'].unique()}")
    
    return measures_df


def display_summary(metadata_df, measures_df):
    """Display summary statistics"""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print("\nMetadata Summary:")
    print(f"  Total deals: {len(metadata_df)}")
    print(f"  Total commitment: ${metadata_df['Commitment'].sum():,.0f}")
    print(f"  Total capital calls: ${metadata_df['Capital Calls'].sum():,.0f}")
    
    if not measures_df.empty:
        print("\nMeasures Summary:")
        print(f"  Total records: {len(measures_df)}")
        print(f"  Unique deals: {measures_df['Deal Name'].nunique()}")
        print(f"  Unique investors: {measures_df['Investor'].nunique()}")
        
        print("\nMeasure breakdown:")
        for measure, count in measures_df['Measure'].value_counts().items():
            print(f"    {measure}: {count}")
    
    print("\nDeals by deployment:")
    for _, row in metadata_df.iterrows():
        if row['Commitment'] > 0:
            deployment = (row['Capital Calls'] / row['Commitment']) * 100
            print(f"  {row['Deal Name']}: {deployment:.1f}% deployed")


def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python pe_data_transformation.py <excel_file>")
        print("\nThis script transforms PE cash flow data from Excel to CSV format")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    
    print(f"Processing: {excel_file}")
    print("="*60)
    
    # Load data
    excel_data = load_excel_data(excel_file)
    
    # Create metadata
    metadata_df = create_metadata(excel_data)
    
    # Create measures
    measures_df = create_measures(excel_data)
    
    # Display summary
    display_summary(metadata_df, measures_df)
    
    # Save to CSV
    metadata_df.to_csv('metadata.csv', index=False)
    measures_df.to_csv('measures.csv', index=False)
    
    print("\n" + "="*60)
    print("OUTPUT FILES CREATED:")
    print("  - metadata.csv")
    print("  - measures.csv")
    print("="*60)


if __name__ == "__main__":
    main()
