#!/usr/bin/env python3
"""
Bonus Assignment: AI-Enhanced PE Deal Descriptions
Generate intelligent descriptions for PE deals using LLM or algorithmic fallback
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Try to import OpenAI if available
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI library not installed. Using algorithmic fallback.")


class PEDescriptionGenerator:
    """Generate AI-enhanced descriptions for PE deals"""
    
    def __init__(self, use_api=False, api_key=None):
        """Initialize the generator with optional API configuration"""
        self.use_api = use_api and HAS_OPENAI
        
        if self.use_api and api_key:
            openai.api_key = api_key
            print("OpenAI API configured")
        elif self.use_api:
            print("Warning: API requested but no key provided")
            self.use_api = False
    
    def calculate_metrics(self, deal_row: pd.Series, measures_df: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate key PE metrics for a deal"""
        metrics = {
            'commitment': float(deal_row.get('Commitment', 0)),
            'capital_calls': float(deal_row.get('Capital Calls', 0)),
            'deployment_pct': 0,
            'distributions': 0,
            'nav': 0,
            'dpi': 0,
            'tvpi': 0
        }
        
        # Calculate deployment percentage
        if metrics['commitment'] > 0 and metrics['capital_calls'] > 0:
            metrics['deployment_pct'] = (metrics['capital_calls'] / metrics['commitment']) * 100
        
        # Calculate metrics from measures if available
        if measures_df is not None and not measures_df.empty:
            deal_name = deal_row.get('Deal Name')
            deal_measures = measures_df[measures_df['Deal Name'] == deal_name]
            
            if not deal_measures.empty:
                # Sum distributions
                dist_measures = deal_measures[deal_measures['Measure'] == 'Distribution']
                if not dist_measures.empty:
                    metrics['distributions'] = dist_measures['Amount'].sum()
                
                # Get latest NAV
                nav_measures = deal_measures[deal_measures['Measure'] == 'Estimated NAV']
                if len(nav_measures) > 0:
                    metrics['nav'] = nav_measures['Amount'].iloc[-1]
                
                # Calculate DPI and TVPI
                if metrics['capital_calls'] > 0:
                    metrics['dpi'] = metrics['distributions'] / metrics['capital_calls']
                    metrics['tvpi'] = (metrics['distributions'] + metrics['nav']) / metrics['capital_calls']
        
        return metrics
    
    def generate_api_description(self, deal_name: str, vintage: int, geography: str, 
                                asset_class: str, metrics: Dict[str, float]) -> Optional[str]:
        """Generate description using OpenAI API"""
        if not self.use_api:
            return None
        
        try:
            prompt = f"""Generate a professional 1-2 line description for this private equity deal:
            
            Deal: {deal_name}
            Vintage: {vintage}
            Geography: {geography}
            Asset Class: {asset_class}
            Commitment: ${metrics['commitment']:,.0f}
            Deployment: {metrics['deployment_pct']:.0f}%
            TVPI: {metrics['tvpi']:.2f}x
            DPI: {metrics['dpi']:.2f}x
            
            Requirements: Maximum 2 sentences, professional tone, focus on key metrics."""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a PE analyst. Write concise, professional deal summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"API error for {deal_name}: {e}")
            return None
    
    def generate_algorithmic_description(self, deal_row: pd.Series, metrics: Dict[str, float]) -> str:
        """Generate description using algorithmic approach"""
        deal_name = str(deal_row.get('Deal Name', 'Unknown'))
        vintage = deal_row.get('Vintage', 'N/A')
        geography = str(deal_row.get('Geography', 'N/A'))
        asset_class = str(deal_row.get('Asset Class', 'P/E'))
        
        description_parts = []
        
        # Main description
        if vintage and vintage != 'N/A' and str(vintage) != 'nan':
            if geography and geography != 'N/A' and geography != 'nan':
                description_parts.append(f"{deal_name} is a {vintage} vintage {asset_class} investment focused on {geography} markets")
            else:
                description_parts.append(f"{deal_name} is a {vintage} vintage {asset_class} investment")
        else:
            description_parts.append(f"{deal_name} is a {asset_class} private equity investment")
        
        # Size and deployment
        if metrics['commitment'] > 0:
            # Format commitment
            if metrics['commitment'] >= 1e9:
                commit_str = f"${metrics['commitment']/1e9:.1f}B"
            elif metrics['commitment'] >= 1e6:
                commit_str = f"${metrics['commitment']/1e6:.0f}M"
            else:
                commit_str = f"${metrics['commitment']:,.0f}"
            
            description_parts.append(f"with {commit_str} in commitments")
            
            if metrics['deployment_pct'] > 0:
                description_parts[-1] += f" ({metrics['deployment_pct']:.0f}% deployed)"
        
        # Performance or status
        if metrics['tvpi'] > 2.5:
            description_parts.append(f"delivering exceptional returns with {metrics['tvpi']:.1f}x TVPI.")
        elif metrics['tvpi'] > 1.8:
            description_parts.append(f"delivering strong returns with {metrics['tvpi']:.1f}x TVPI.")
        elif metrics['dpi'] > 1.0:
            description_parts.append(f"generating solid distributions with {metrics['dpi']:.1f}x DPI.")
        elif metrics['deployment_pct'] > 90:
            description_parts.append("currently in harvest phase.")
        elif metrics['deployment_pct'] > 50:
            description_parts.append("actively deploying capital.")
        elif metrics['deployment_pct'] > 0:
            description_parts.append("in early investment period.")
        else:
            description_parts.append("in fundraising stage.")
        
        return " ".join(description_parts[:3])  # Limit to 3 parts max
    
    def generate_description(self, deal_row: pd.Series, measures_df: pd.DataFrame = None) -> str:
        """Generate AI-enhanced description for a PE deal"""
        # Calculate metrics
        metrics = self.calculate_metrics(deal_row, measures_df)
        
        # Try API first if configured
        if self.use_api:
            deal_name = str(deal_row.get('Deal Name', 'Unknown'))
            vintage = deal_row.get('Vintage', None)
            geography = str(deal_row.get('Geography', 'N/A'))
            asset_class = str(deal_row.get('Asset Class', 'P/E'))
            
            api_description = self.generate_api_description(
                deal_name, vintage, geography, asset_class, metrics
            )
            
            if api_description:
                return api_description
        
        # Fall back to algorithmic generation
        return self.generate_algorithmic_description(deal_row, metrics)


def load_data(metadata_file='metadata.csv', measures_file='measures.csv'):
    """Load metadata and measures data"""
    try:
        metadata_df = pd.read_csv(metadata_file)
        print(f"Loaded metadata: {len(metadata_df)} deals")
    except FileNotFoundError:
        print(f"Error: {metadata_file} not found")
        sys.exit(1)
    
    try:
        measures_df = pd.read_csv(measures_file)
        print(f"Loaded measures: {len(measures_df)} records")
    except FileNotFoundError:
        print(f"Warning: {measures_file} not found. Continuing without measures.")
        measures_df = pd.DataFrame()
    
    return metadata_df, measures_df


def display_results(metadata_df):
    """Display enhanced metadata with AI descriptions"""
    print("\n" + "="*80)
    print("PE DEALS WITH AI DESCRIPTIONS")
    print("="*80)
    
    for idx, row in metadata_df.iterrows():
        print(f"\n{idx+1}. {row['Deal Name']}")
        print("-"*40)
        print(f"   Vintage: {row.get('Vintage', 'N/A')}")
        print(f"   Geography: {row.get('Geography', 'N/A')}")
        print(f"   Asset Class: {row.get('Asset Class', 'N/A')}")
        
        commitment = row.get('Commitment', 0)
        capital_calls = row.get('Capital Calls', 0)
        
        if pd.notna(commitment) and commitment > 0:
            print(f"   Commitment: ${commitment:,.0f}")
            if pd.notna(capital_calls) and capital_calls > 0:
                deployment = (capital_calls / commitment) * 100
                print(f"   Deployment: {deployment:.1f}%")
        
        print(f"\n   AI Description:")
        print(f"   {row['AI Description']}")
    
    print("\n" + "="*80)


def main():
    """Main execution function"""
    print("PE DEAL DESCRIPTION GENERATOR")
    print("="*60)
    
    # Check for API key in environment
    api_key = os.environ.get('OPENAI_API_KEY')
    use_api = bool(api_key)
    
    if use_api:
        print("OpenAI API key detected")
    else:
        print("No API key found. Using algorithmic descriptions.")
    
    # Load data
    metadata_df, measures_df = load_data()
    
    # Initialize generator
    generator = PEDescriptionGenerator(use_api=use_api, api_key=api_key)
    
    # Generate descriptions
    print("\nGenerating AI descriptions...")
    print("-"*40)
    
    descriptions = []
    for idx, row in metadata_df.iterrows():
        deal_name = row['Deal Name']
        print(f"Processing {idx+1}/{len(metadata_df)}: {deal_name}")
        
        description = generator.generate_description(row, measures_df)
        descriptions.append(description)
    
    # Add descriptions to dataframe
    metadata_df['AI Description'] = descriptions
    
    # Display results
    display_results(metadata_df)
    
    # Calculate statistics
    print("\nSummary Statistics:")
    print(f"  Total deals: {len(metadata_df)}")
    print(f"  Average description length: {metadata_df['AI Description'].str.len().mean():.0f} characters")
    print(f"  Using API: {'Yes' if use_api else 'No (Algorithmic)'}")
    
    # Save enhanced metadata
    output_file = 'metadata_with_ai_descriptions.csv'
    metadata_df.to_csv(output_file, index=False)
    print(f"\nSaved enhanced metadata to: {output_file}")
    
    # Save sample for documentation
    with open('sample_descriptions.txt', 'w') as f:
        f.write("SAMPLE AI-GENERATED DEAL DESCRIPTIONS\n")
        f.write("="*50 + "\n\n")
        for _, row in metadata_df.head(3).iterrows():
            f.write(f"Deal: {row['Deal Name']}\n")
            f.write(f"Description: {row['AI Description']}\n\n")
    
    print("Saved sample descriptions to: sample_descriptions.txt")


if __name__ == "__main__":
    main()
