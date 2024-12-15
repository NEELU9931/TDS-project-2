# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "pandas",
#     "seaborn",
#     "matplotlib",
#     "uvicorn",
#     "openai",
#     "requests"
# ]
# ///

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import requests
import json
import numpy as np

# Initialize OpenAI API key from environment variable
openai.api_key = os.environ["AIPROXY_TOKEN"]

def load_data(filename):
    """Load CSV data."""
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_data(data):
    """Perform basic analysis on the dataset."""
    # Summary statistics
    summary_stats = data.describe(include='all')
    
    # Missing values
    missing_values = data.isnull().sum()
    
    # Correlation matrix
    correlation_matrix = data.corr()
    
    # Check for outliers (using Z-score method)
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers = (z_scores > 3).sum()
    
    # Return all analysis results
    return summary_stats, missing_values, correlation_matrix, outliers

def create_correlation_heatmap(correlation_matrix, output_filename):
    """Create a heatmap for correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f')
    plt.title("Correlation Matrix Heatmap")
    plt.savefig(output_filename)
    plt.close()

def send_to_llm(data_description, analysis_summary):
    """Send the analysis results to LLM for summary and insights."""
    prompt = f"""
    Dataset Description: {data_description}
    Analysis Summary: {analysis_summary}
    Based on this, provide insights, implications, and recommendations.
    """
    
    # Sending prompt to OpenAI GPT-4o-Mini model
    response = openai.Completion.create(
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=500
    )
    
    return response.choices[0].text.strip()

def generate_markdown_report(data, summary_stats, missing_values, correlation_matrix, insights, charts):
    """Generate the Markdown README report with analysis results."""
    report = f"""
    # Data Analysis Report
    
    ## Dataset Overview
    - Number of Rows: {len(data)}
    - Number of Columns: {data.shape[1]}
    
    ## Summary Statistics
    {summary_stats}
    
    ## Missing Values
    {missing_values}
    
    ## Outliers
    Number of Outliers detected: {sum(outliers > 0)}

    ## Correlation Matrix
    ![Correlation Heatmap](./{charts[0]})
    
    ## Insights and Implications
    {insights}
    """
    
    with open("README.md", "w") as file:
        file.write(report)

def main(filename):
    """Main function to run the script."""
    
    # Load dataset
    data = load_data(filename)
    if data is None:
        return
    
    # Perform analysis
    summary_stats, missing_values, correlation_matrix, outliers = analyze_data(data)
    
    # Create correlation heatmap
    create_correlation_heatmap(correlation_matrix, "correlation_heatmap.png")
    
    # Send data description and analysis summary to LLM for insights
    data_description = f"Columns: {', '.join(data.columns)}"
    analysis_summary = f"Summary Stats: {summary_stats.head()} Missing Values: {missing_values.head()}"
    insights = send_to_llm(data_description, analysis_summary)
    
    # Generate Markdown report
    generate_markdown_report(data, summary_stats, missing_values, correlation_matrix, insights, ["correlation_heatmap.png"])

if __name__ == "__main__":
    # Ensure a CSV filename is passed as argument
    import sys
    if len(sys.argv) < 2:
        print("Please provide a CSV filename.")
    else:
        main(sys.argv[1])






