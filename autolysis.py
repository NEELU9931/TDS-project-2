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
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Install necessary libraries if not already installed
try:
    import openai
except ImportError:
    os.system("pip install openai")
    import openai

try:
    from IPython.display import Markdown, display
except ImportError:
    os.system("pip install IPython")
    from IPython.display import Markdown, display


def analyze_and_narrate(csv_filename):
    try:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        return "Error: CSV file not found."

    # Basic data exploration and visualization (replace with more sophisticated analysis)

    # 1. Descriptive statistics
    description = df.describe(include='all')

    # 2. Correlation Matrix (if applicable)
    try:
        numeric_cols = df.select_dtypes(include=['number'])
        if not numeric_cols.empty:
            correlation_matrix = numeric_cols.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix")
            plt.savefig("correlation_matrix.png")
    except:
        print("No numeric columns found for correlation matrix")

    # 3. Histogram or Count Plot (choose based on data)
    try:
        first_column = df.columns[0]
        if pd.api.types.is_numeric_dtype(df[first_column]):
            plt.figure(figsize=(8, 6))
            sns.histplot(df[first_column], kde=True)
            plt.title(f"Distribution of {first_column}")
            plt.savefig("histogram.png")
        else:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=first_column, data=df)
            plt.title(f"Count of {first_column}")
            plt.xticks(rotation=45, ha='right')
            plt.savefig("countplot.png")
    except:
        print("Error generating plot")

    # Generate narrative using an LLM
    openai.api_key = os.environ["AIPROXY_TOKEN"]

    prompt = f"""Analyze the following data and create a short story based on it.
    Data Description:\n{description.to_string()}

    The data is related to: {csv_filename}

    Narrative:"""


    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=prompt,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )
    narrative = response.choices[0].text.strip()


    # Create README.md
    readme_content = f"# Automated Data Analysis of {csv_filename}\n\n{narrative}\n\n"

    # Add images to README.md
    if os.path.exists("correlation_matrix.png"):
        readme_content += "![](correlation_matrix.png)\n\n"
    if os.path.exists("histogram.png"):
        readme_content += "![](histogram.png)\n\n"
    if os.path.exists("countplot.png"):
        readme_content += "![](countplot.png)\n\n"
    
    with open("README.md", "w") as f:
        f.write(readme_content)

    return "Analysis complete. Check README.md"

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <csv_filename>")
        sys.exit(1)
    csv_filename = sys.argv[1]
    result = analyze_and_narrate(csv_filename)
result
