# Titanic Data Analysis Project

## Project Overview

This project contains various scripts and data files related to the analysis of the Titanic dataset. The purpose of the project is to explore and analyze the Titanic dataset using different tools and methodologies, including optimization techniques (e.g., Gurobi), subgroup discovery algorithms (e.g., PySubgroup), and visualization of results.

## Files Included

### 1. `titanic.csv`

- **Description**: This file contains the Titanic dataset with various features such as passenger details (age, gender, ticket class, etc.) and survival information.
- **Usage**: It is used as the primary data source for all analysis and experimentation in the project.

### 2. `titanic.ipynb`

- **Description**: A Jupyter Notebook for performing exploratory data analysis (EDA) on the Titanic dataset.
- **Key Features**:
  - Data loading and cleaning.
  - Exploratory data analysis using descriptive statistics and visualizations.
  - Explanations of the dataset and feature selection.
- **Recommendation**: Consider renaming this notebook to something like `titanic_analysis.ipynb` for clarity.

### 3. `results.ipynb`

- **Description**: A Jupyter Notebook containing the results of the optimization and subgroup discovery processes.
- **Key Features**:
  - Comparison of objective values (e.g., WRAcc) obtained from Gurobi and PySubgroup.
  - Visualizations showing how the results change with varying dataset sizes.

### 4. `titanic.html`

- **Description**: An exported HTML version of the EDA
- **Usage**: A read-only file for sharing or presenting the findings without requiring a Jupyter environment.

### 5. `run_gurobi.py`

- **Description**: A Python script implementing subgroup discovery using the Gurobi optimizer.
- **Key Features**:
  - Handles the exact optimization of subgroup discovery problems.
  - Requires Gurobi installation and license.
- **Usage**: Execute this script to perform subgroup discovery using Gurobi.

### 6. `run_pysubgroup.py`

- **Description**: A Python script implementing subgroup discovery using the PySubgroup library.
- **Key Features**:
  - Implements a depth-first search (DFS) for identifying subgroups.
  - Optimized for efficient computation using pruning techniques.
- **Usage**: Execute this script to perform subgroup discovery using PySubgroup.

### 7. `pdf.code-profile`

- **Description**: A configuration or profiling file (potentially related to code execution or optimization).
- **Usage**: Details about this file need clarification, as its purpose is unclear from the filename.

## Getting Started

1. **Install Dependencies**:

   - Install Python and Jupyter Notebook.
   - Install required libraries (e.g., Pandas, Matplotlib, PySubgroup, and Gurobi).
   - For Gurobi, ensure you have a valid license and installation.

2. **Run the Analysis**: (e.g., WRAcc) and comparisons between Gurobi and PySubgroup methods.

## Notes

- Ensure that all dependencies are properly installed before running the scripts.
- The `titanic.csv` dataset is the foundation for all analyses; do not modify it unless absolutely necessary.


