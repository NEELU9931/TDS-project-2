
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///


import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import logging

# Configure logging
logging.basicConfig(
    filename="analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize OpenAI
openai_api_key = os.getenv("AIPROXY_TOKEN")
if not openai_api_key:
    logging.error("AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)
openai.api_key = openai_api_key

def load_data(file_path):
    """Load the dataset from a CSV file."""
    encodings = ["utf-8", "ISO-8859-1", "Windows-1252"]
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            logging.info(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns using {encoding} encoding.")
            return data
        except UnicodeDecodeError:
            logging.warning(f"Encoding {encoding} failed. Trying next...")
        except Exception as e:
            logging.error(f"Error loading data with {encoding} encoding: {e}")
    logging.error("Unable to load the dataset with tried encodings.")
    sys.exit(1)

def basic_analysis(data):
    """Perform basic statistical analysis."""
    analysis = {
        "Summary Statistics": data.describe(include='all').to_dict(),
        "Missing Values": data.isnull().sum().to_dict(),
        "Data Types": data.dtypes.apply(str).to_dict(),
    }
    return analysis

def extended_analysis(data):
    """Perform additional data analysis."""
    outlier_info = {}
    for col in data.select_dtypes(include=["float64", "int64"]).columns:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = data[(data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)]
        outlier_info[col] = len(outliers)

    unique_values = {col: data[col].nunique() for col in data.select_dtypes(include=["object"]).columns}
    return {
        "Outlier Information": outlier_info,
        "Unique Values": unique_values
    }

def visualize_correlation(data, output_dir, dataset_name):
    """Generate a correlation heatmap."""
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    if numeric_data.empty:
        logging.warning("No numeric data available for correlation analysis.")
        return
    corr = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    output_file = os.path.join(output_dir, f"{dataset_name}_heatmap.png")
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Correlation heatmap saved to {output_file}.")

def visualize_distributions(data, output_dir, dataset_name):
    """Generate distribution plots."""
    for col in data.select_dtypes(include=["float64", "int64"]).columns:
        plt.figure()
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        output_file = os.path.join(output_dir, f"{dataset_name}_{col}_distribution.png")
        plt.savefig(output_file)
        plt.close()
        logging.info(f"Distribution plot saved to {output_file}.")

def query_llm(prompt):
    """Query OpenAI for insights."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        return ""

def narrate_story(data_info, basic_analysis, extended_analysis, insights):
    """Generate a narrative from the analysis."""
    prompt = f"""
    You are a data analyst assistant. Write a Markdown story for the following data analysis:
    Dataset Information:
    {data_info}
    
    Basic Analysis:
    {basic_analysis}
    
    Extended Analysis:
    {extended_analysis}
    
    Key Insights:
    {insights}
    
    Please include headers and structure it well for a README.md file.
    """
    return query_llm(prompt)

def save_story(story, output_dir):
    """Save the narrative to a README.md file."""
    output_file = os.path.join(output_dir, "README.md")
    with open(output_file, "w") as f:
        f.write(story)
    logging.info(f"Narrative saved to {output_file}.")

def main():
    if len(sys.argv) != 2:
        logging.error("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(os.getcwd(), dataset_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    data = load_data(file_path)

    # Basic analysis
    analysis = basic_analysis(data)
    extended = extended_analysis(data)

    # Visualizations
    visualize_correlation(data, output_dir, dataset_name)
    visualize_distributions(data, output_dir, dataset_name)

    # Generate story
    data_info = {
        "Columns": data.columns.tolist(),
        "Shape": data.shape,
    }
    insights = "The analysis provided insights into data patterns, correlations, and outliers."
    story = narrate_story(data_info, analysis, extended, insights)

    # Save results
    save_story(story, output_dir)

if __name__ == "__main__":
    main()
