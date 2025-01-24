import pysubgroup as ps
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import memory_usage
import time
import os
from datetime import datetime

# Constants for subgroup discovery
TARGET_VALUE = 1  # Target for 'Survived' status
RESULT_SET_SIZE = 1000
SEARCH_DEPTH = 400
LOG_DIR = "/Users/claragazer/Desktop/pysubgroup_results"  # Ensure this path is correct

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

import pysubgroup as ps
import numpy as np
import pysubgroup as ps
import numpy as np

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Titanic data for subgroup discovery.
    """
    try:
        pd.set_option('future.no_silent_downcasting', True)
        df_full = pd.read_csv(file_path)
        df = df_full.sample(n=9000, random_state=42)

        # Encode categorical columns
        df['Gender'] = df['Gender'].replace({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})

        # Fill missing values for Age and Fare with the mean
        df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill with column mean
        df['Fare'] = df['Fare'].fillna(df['Fare'].mean())  # Fill with column mean

        # Handle Gender and Embarked with mode (most common value)
        df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0]).astype(int)
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]).astype(int)

        # Preprocess Cabin
        df['Cabin'] = df['Cabin'].fillna('U')  # Replace NaN with 'U'
        df['Cabin'] = df['Cabin'].str[0]  # Extract first letter of Cabin
        print("Typ von df:", type(df))  # Zeigt den Typ von df an

        print(f"Anzahl der Zeilen: {len(df)}")

        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


def define_target(df):
    """
    Define the target for subgroup discovery based on survival status.
    """
    target = ps.BinaryTarget('Survived', TARGET_VALUE)
    return target

def measure_memory_usage(func, *args, **kwargs):
    """
    Measures memory usage of a given function.
    """
    mem_usage, result = memory_usage((func, args, kwargs), interval=0.1, retval=True)
    peak_memory = max(mem_usage)
    print(f"Peak memory usage of {func.__name__}: {peak_memory:.4f} MB")
    
    return result, peak_memory

def define_selectors(data):
    """
    Define selectors for numerical and categorical columns with explicit bounds.
    """
    selectors = []

    # Define interval selectors for numerical columns
    for column in ["Age", "Fare"]:
        q25 = data[column].quantile(0.25)
        q75 = data[column].quantile(0.75)
        selectors.append(ps.IntervalSelector(column, float("-inf"), q25))
        selectors.append(ps.IntervalSelector(column, q75, float("inf")))
    for column in ["SibSp", "Parch"]:
        try:
            # Konvertiere alle Werte in den Spalten zu numerischen Typen
            data[column] = pd.to_numeric(data[column], errors='coerce') # important so that we don't have 35 but 23 selectors
            unique_values = data[column].dropna().unique()
            for value in unique_values:
                selectors.append(ps.EqualitySelector(column, value))
        except Exception as e:
            print(f"Error while creating selectors for column {column}: {e}")
    
    # Define nominal selectors for categorical columns
    selectors.extend([
        ps.EqualitySelector("Gender", 0),
        ps.EqualitySelector("Gender", 1),
        ps.EqualitySelector("Pclass", 1),
        ps.EqualitySelector("Pclass", 2),
        ps.EqualitySelector("Pclass", 3),
    ])


    # selectors.append(ps.EqualitySelector("Parch", 6))

    print("LENGTH")
    print(len(selectors))
    print(selectors)
    return selectors

def run_subgroup_discovery(df, target, search_space):
    quality_function = ps.WRAccQF()
    dfs = ps.DFS()
    task = ps.SubgroupDiscoveryTask(
        df, target, search_space, result_set_size=RESULT_SET_SIZE, depth=SEARCH_DEPTH, qf=quality_function
    )
    result = dfs.execute(task)
    return result.to_dataframe()

def save_run_results(result_df, runtime, peak_memory, iteration, file):
    """
    Save the results of a single run to an open file.
    """
    total_memory_usage = peak_memory
    total_runtime = runtime

    if result_df.empty:
        print(f"Warning: result_df is empty for iteration {iteration}. No data to write.")
        file.write(f"\nIteration {iteration}: No data to write.\n")
        return

    # Schreibe die Ergebnisse der aktuellen Iteration in die Datei
    file.write(f"\nIteration {iteration} Results:\n")
    file.write(f"Runtime: {runtime:.2f} seconds\n")
    file.write(f"Peak Memory Usage: {peak_memory:.2f} MB\n")

    # Iteriere durch die Subgruppen und schreibe die Details
    for i, row in result_df.iterrows():
        file.write(f"\nSubgroup {i + 1}:\n")
        file.write(f"  Conditions: {row['subgroup']}\n")
        file.write(f"  Quality Score (Objective): {row['quality']:.4f}\n")
        file.write(f"  Target Share: {row['target_share_sg']:.2f}\n")
    
    # Durchschnittswerte pro Subgruppe
    avg_runtime = total_runtime / len(result_df) if len(result_df) > 0 else 0
    avg_memory_usage = total_memory_usage / len(result_df) if len(result_df) > 0 else 0

    file.write(f"\nAverage Runtime per Subgroup: {avg_runtime:.2f} seconds\n")
    file.write(f"Average Memory Usage per Subgroup: {avg_memory_usage:.2f} MB\n")
    file.write("\n" + "-"*40 + "\n")

def main(file_path):
    """
    Main function to run the PySubgroup algorithm on the dataset and log memory and runtime.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path_results = os.path.join(LOG_DIR, f"subgroup_run_{timestamp}.txt")
    
    try:
        # Öffne die Datei nur einmal für alle Iterationen
        with open(file_path_results, "w") as f:
            f.write("Subgroup Discovery Results (from pysubgroup)\n")
            f.write(f"File: {file_path}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            # Schleife, um das Experiment 10 Mal durchzuführen
            for iteration in range(1, 3):
                start_time = time.time()

                # Messe den Speicherverbrauch und erhalte das Ergebnis
                mem_usage, (result_df, df) = memory_usage((discovery_task, (file_path,)), retval=True, interval=0.1)
                peak_memory = max(mem_usage)  # Maximaler Speicherverbrauch in MB

                # Berechne die Laufzeit
                runtime = time.time() - start_time

                # Speichere die Ergebnisse in die geöffnete Datei
                save_run_results(result_df, runtime, peak_memory, iteration, f)

            print(f"Results saved to {file_path_results}")

    except Exception as e:
        print(f"An error occurred during the main execution: {e}")

def discovery_task(file_path):
    """
    Run the subgroup discovery process for a given dataset file path.
    """
    df = load_and_preprocess_data(file_path)
    target = define_target(df)
    search_space = define_selectors(df)
    result_df = run_subgroup_discovery(df, target, search_space)
    return result_df, df

# Execute main function with the specified file path
if __name__ == "__main__":
    file_path = "/Users/claragazer/Desktop/Bachelorarbeit/Bachelor2/data/titanic.csv"  # Adjust path as necessary
    main(file_path)
