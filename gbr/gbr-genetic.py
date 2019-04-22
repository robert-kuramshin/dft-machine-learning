import random 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import StratifiedKFold
from genetic_selection import GeneticSelectionCV

def main():
    #read data
    test = pd.read_csv("../data/test.csv")
    train = pd.read_csv("../data/train.csv")


    #specify feature column names
    feature_cols = [
    'Radius A [ang]',
    'Radius B [ang]',
    'Formation energy [eV/atom]',
    'Stability [eV/atom]',
    'Magnetic moment [mu_B]',
    'Volume per atom [A^3/atom]',
    'a [ang]',
    'b [ang]',
    'c [ang]',
    'alpha [deg]',
    'beta [deg]',
    'gamma [deg]',
    'Vacancy energy [eV/O atom]',
    'Octahedral Factor',
    'Tolerance Factor',
    'A Ionization Energy',
    'B Ionization Energy',
    'A Electronegativity',
    'B Electronegativity',
    "A rs",
    "A rp",
    "A rd",
    "B rs",
    "B rp",
    "B rd",
    "B EA",
    "B s total",
    "B p total",
    "B d total",
    "B f total",
    "A g",
    "A p",
    "B g",
    "B p"
    ]


    feature_names = [
    'Radius A [ang]',
    'Radius B [ang]',
    'Formation energy [eV/atom]',
    'Stability [eV/atom]',
    'Magnetic moment [mu_B]',
    'Volume per atom [A^3/atom]',
    'a [ang]',
    'b [ang]',
    'c [ang]',
    'alpha [deg]',
    'beta [deg]',
    'gamma [deg]',
    'Vacancy energy [eV/O atom]',
    'Octahedral Factor',
    'Tolerance Factor',
    'A Ionization Energy',
    'B Ionization Energy',
    'A Electronegativity',
    'B Electronegativity',
    'Goldschmidt Tolerance Factor'
    ]

    #splitting into dependant and independant variables
    X_train = train.loc[:, feature_cols]
    y_train = train["Band gap [eV]"]

    X_test = test.loc[:, feature_cols]
    y_test = test["Band gap [eV]"]
    paramgrid = {'n_estimators': [1500,5000],
                        'max_depth': np.logspace(0, 1, num=10, base=10),
                        'min_samples_leaf': np.logspace(0, 1, num=10, base=10),
                        'min_samples_split': np.logspace(-1, 1, num=20, base=10),
                        'learning_rate': np.logspace(-1, 0, num=10, base=10),
                    'loss': ['ls','lad','huber']}

    selector = GeneticSelectionCV(estimator=GradientBoostingRegressor(),
                                    cv=5,
                                    verbose=1,
                                    scoring="accuracy",
                                    max_features=5,
                                    n_population=50,
                                    crossover_proba=0.5,
                                    mutation_proba=0.2,
                                    n_generations=40,
                                    crossover_independent_proba=0.5,
                                    mutation_independent_proba=0.05,
                                    tournament_size=3,
                                    n_gen_no_change=10,
                                    caching=True,
                                    n_jobs=-1)
    selector = selector.fit(X_train, y_train)

    print(selector.support_)

if __name__ == "__main__":
    main()

