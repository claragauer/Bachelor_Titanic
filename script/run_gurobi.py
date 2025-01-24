import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
from datetime import datetime
import time
import os

# Constants
THETA_DC = 5            # Maximum number of selectors allowed
THETA_CC = 10           # Minimum coverage required for the subgroup
THETA_MAX_RATIO = 0.7   # Maximum ratio of cases that can be included in the subgroup
AMOUNT_OF_RUNS = 1

LOG_DIR = "/Users/claragazer/Desktop/gurobi_results"
os.makedirs(LOG_DIR, exist_ok=True)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess Titanic dataset by encoding categorical variables and handling missing values.
    """
    try:
        # Load dataset
        
        df = pd.read_csv(file_path, nrows=1000) #CG: evtl 0-1000, dann 1000-3000 usw 
        print(f"Anzahl der Zeilen im DataFrame: {df.shape[0]}")

        # Encode categorical columns
        df['Gender'] = df['Gender'].replace({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})
        df['Age'] = pd.to_numeric(df['Age']) 

        # Fill missing values for Age and Fare with the mean
        df['Age'] = df['Age'].fillna(df['Age'].mean())  # Fill with column mean
        df['Fare'] = df['Fare'].fillna(df['Fare'].mean())  # Fill with column mean

        # Handle Gender and Embarked with mode (most common value)
        df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0]).astype(int)
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]).astype(int)

        # Preprocess Cabin
        df['Cabin'] = df['Cabin'].fillna('U')  # Replace NaN with 'U'
        df['Cabin'] = df['Cabin'].str[0]  # Extract first letter of Cabin

        print("Data after preprocessing:")
        print(df.head())
        print(df.dtypes)

        return df
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


def define_selectors(data):
    """
    Define selectors based on quantiles for numerical columns.
    """
    selectors = {}
      # Define high and low selectors using quantiles for numerical columns
    for column in ["Age", "Fare"]:
        threshold_low = float(data[column].quantile(0.25))
        threshold_high = float(data[column].quantile(0.75))
        selectors[f"High_{column}"] = data[column] >= threshold_high
        selectors[f"Low_{column}"] = data[column]< threshold_low

    # Add selectors for categorical columns
    selectors["Male"] = (data["Gender"] == 0).astype(int)
    selectors["Female"] = (data["Gender"] == 1).astype(int)
    
    selectors["Pclass_1"] = (data["Pclass"] == 1).astype(int)
    selectors["Pclass_2"] = (data["Pclass"] == 2).astype(int)
    selectors["Pclass_3"] = (data["Pclass"] == 3).astype(int)

    # Parch conditions
    selectors["Parch_2"] = (data["Parch"] == 2).astype(int)
    selectors["Parch_0"] = (data["Parch"] == 0).astype(int)
    selectors["Parch_1"] = (data["Parch"] == 1).astype(int)
    selectors["Parch_5"] = (data["Parch"] == 5).astype(int)
    selectors["Parch_3"] = (data["Parch"] == 3).astype(int)
    selectors["Parch_6"] = (data["Parch"] == 6).astype(int)
    selectors["Parch_4"] = (data["Parch"] == 4).astype(int)

    # SibSp conditions
    selectors["SibSp_5"] = (data["SibSp"] == 5).astype(int)
    selectors["SibSp_1"] = (data["SibSp"] == 1).astype(int)
    selectors["SibSp_0"] = (data["SibSp"] == 0).astype(int)
    selectors["SibSp_4"] = (data["SibSp"] == 4).astype(int)
    selectors["SibSp_2"] = (data["SibSp"] == 2).astype(int)
    selectors["SibSp_3"] = (data["SibSp"] == 3).astype(int)
    selectors["SibSp_8"] = (data["SibSp"] == 8).astype(int)
    
    print(selectors)
    print("LENGTH")
    print(len(selectors))
    return selectors
  
def setup_model(n_cases, selectors):
    model = gp.Model("Titanic_Subgroup_Discovery")
    model.setParam('OutputFlag', 1)
    
    T = model.addVars(n_cases, vtype=GRB.BINARY, name="T")
    D = model.addVars(len(selectors), vtype=GRB.BINARY, name="D")

    # Get selector keys and values
    selector_keys = list(selectors.keys())  # List of selector names
    selector_values = list(selectors.values())  # List of binary Pandas Series

    # Add constraints to link T[c] with D[i] and selectors
    for c in range(n_cases):
        # Ensure T[c] is only 1 if at least one active selector satisfies the condition
        model.addConstr(
            T[c] <= gp.quicksum(D[i] * selector_values[i].iloc[c] for i in range(len(selectors))),
            f"SubgroupCondition_{c}"
        )
    
    # C8
    # Ensure T[c] = 0 if no selectors are active
    for c in range(n_cases):
        model.addConstr(
            T[c] >= gp.quicksum(D[i] * selector_values[i].iloc[c] for i in range(len(selectors))) / len(selectors),
            f"SubgroupCondition_{c}_NonZero"
        )

    # Add constraints for maximum number of active selectors
    add_constraints(model, n_cases, selectors, T, D)

    # Zusätzliche Constraints
    add_constraints(model, n_cases, selectors, T, D)
    return model, T, D


def add_constraints(model, n_cases, selectors, T, D):
    """
    Add constraints to the Gurobi model to enforce conditions like maximum selectors, 
    minimum and maximum subgroup sizes, and subgroup non-triviality.
    """
    model.addConstr(gp.quicksum(D[i] for i in range(len(selectors))) <= THETA_DC, "MaxSelectors")
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) >= THETA_CC, "MinCases")
    theta_max = int(THETA_MAX_RATIO * n_cases)
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) <= theta_max, "MaxSize")
    model.addConstr(gp.quicksum(T[c] for c in range(n_cases)) >= 1, "NonEmptySubgroup")
    model.addConstr(gp.quicksum(D[i] for i in range(len(selectors))) >= 1, "AtLeastOneSelector")


def set_objective(model, T, data, n_cases):
    # Positives: Überlebende
    positives = data["Survived"].astype(int)
    n_cases = len(data)
    global_target_share = positives.sum() / n_cases

    # Größe der Subgruppe (alle Fälle, wo T[i] = 1)
    size_SG = gp.quicksum(T[i] for i in range(n_cases))
    positives_SG = gp.quicksum(T[i] * positives.iloc[i] for i in range(n_cases))

    # Definition von PosRatio (Anteil der Überlebenden in der Subgruppe)
    PosRatio = model.addVar(name="PosRatio", lb=0, ub=1, vtype=GRB.CONTINUOUS)
    model.addConstr(PosRatio * size_SG == positives_SG, "PosRatio_Definition")


    # WRAcc-Formel als Ziel
    wracc = (size_SG / n_cases) * (PosRatio - global_target_share)
    model.setObjective(wracc, GRB.MAXIMIZE)

    # **Nebenbedingung:** Alle ausgewählten Mitglieder (T[i] = 1) müssen \( Survived = 1 \) haben
    for i in range(n_cases):
        model.addConstr(T[i] <= positives.iloc[i], f"Constraint_Survived_{i}")

def set_objective_linearized(model, T, data, n_cases):
    positives = data["Survived"].astype(int)
    global_target_share = positives.mean()

    # Größe des Subsets und positive Fälle im Subset
    size_SD = gp.quicksum(T[i] for i in range(n_cases))
    positives_SD = gp.quicksum(T[i] * positives.iloc[i] for i in range(n_cases))

    # Neue Variable für das Produkt (size_SD * PosRatio)
    Z = model.addVar(lb=0, ub=n_cases, name="Z")
    PosRatio = model.addVar(lb=0, ub=1, name="PosRatio")

    # Nebenbedingungen für Z = PosRatio * size_SD
    model.addConstr(Z <= size_SD, "Z_UpperBound1")
    model.addConstr(Z <= n_cases * PosRatio, "Z_UpperBound2")
    model.addConstr(Z >= size_SD - n_cases * (1 - PosRatio), "Z_LowerBound")

    # Verknüpfung von Z mit positives_SD
    model.addConstr(Z == positives_SD, "Z_Definition")

    # Vollständig linearisierte Zielfunktion
    wracc = (Z / n_cases) - (size_SD / n_cases) * global_target_share
    model.setObjective(wracc, GRB.MAXIMIZE)


def run_optimization(model):
    """
    Run the optimization process and display the results if an optimal solution is found.
    """
    model.setParam('TimeLimit', 2000)  # 300: 5 Minuten, 2000: 33 Minuten 
    model.setParam('Presolve', 2)  # Nutzt eine intensivere Vorverarbeitung
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")

    else:
        print(f"Optimization was unsuccessful. Gurobi status code: {model.status}")

def measure_memory_usage(func, *args, **kwargs):
    """
    Measures memory usage of a given function.
    """
    mem_usage, result = memory_usage((func, args, kwargs), interval=0.1, retval=True)
    peak_memory = max(mem_usage)
    print(f"Peak memory usage of {func.__name__}: {peak_memory:.4f} MB")
    
    return result, peak_memory


def execute_optimization(file_path):
    # Load data
    data = load_and_preprocess_data(file_path)
    selectors = define_selectors(data)
    n_cases = len(data)

    # Set up and run optimization in one function
    model, T, D = setup_model(n_cases, selectors)
    
    set_objective_linearized(model, T, data, n_cases)
    run_optimization(model)

    # Extract and check optimality
    is_optimal = model.status == GRB.OPTIMAL
    if is_optimal:
        wracc = model.objVal
        gap = model.MIPGap
        active_selectors = [
            name for i, name in enumerate(selectors.keys())
            if D[i].x > 0.5  # Access the solution directly
        ]
    else:
        wracc, gap, active_selectors = None, None, []
    

    # Anzahl der positiven Fälle in der Subgruppe (manuell)
    #manual_subgroup_positives = sum((data["Parch"] == 6) & (data["Survived"] == 1))

    # Vergleich mit positives_SD
    #print(f"Manual subgroup survivors: {manual_subgroup_positives}")

    
    return {
        "wracc": wracc,
        "gap": gap,
        "is_optimal": is_optimal,
        "active_selectors": active_selectors,
        "data": data
    }

def save_run_results(run_results):

    LOG_DIR = "/Users/claragazer/Desktop/gurobi_results"
    os.makedirs(LOG_DIR, exist_ok=True)  
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(LOG_DIR, f"gurobi_subgroup_run_{timestamp}.txt")

    with open(file_path, "w") as f:
        f.write("Gurobi Subgroup Discovery Results (Multiple Runs)\n\n")
        for run in run_results:
            f.write(f"Run {run['run_number']}:\n")
            f.write(f"  Runtime: {run['runtime']:.2f} seconds\n")
            f.write(f"  Peak Memory Usage: {run['peak_memory']:.2f} MB\n")
            f.write(f"  Gap: {run['gap']}\n")
            f.write(f"  WRAcc Score: {run['wracc']}\n")
            f.write(f"  Optimal Solution: {run['is_optimal']}\n")
            f.write(f"  Active Selectors: {run['active_selectors']}\n\n")
    print(f"Results saved to {file_path}")

def main(file_path):
    run_results = []
    for run in range(AMOUNT_OF_RUNS):
        start_time = time.time()
        mem_usage, result = memory_usage(
            (execute_optimization, (file_path,)), retval=True, interval=0.1
        )
        peak_memory = max(mem_usage)
        runtime = time.time() - start_time

        run_results.append({
            "run_number": run + 1,
            "runtime": runtime,
            "peak_memory": peak_memory,
            "gap": result["gap"],
            "wracc": result["wracc"],
            "is_optimal": result["is_optimal"],
            "active_selectors": result["active_selectors"]
        })

    save_run_results(run_results)

if __name__ == "__main__":
    file_path = "/Users/claragazer/Desktop/Bachelorarbeit/Bachelor2/data/titanic.csv"
    main(file_path)