import numpy as np


def edge_constraints_encoding(constraint):
    all_constraints = ["Conflicts", "Collocation", "OneToManyDependency", "ExclusiveDeployment", "UpperBound",
                       "LowerBound", "EqualBound"]
    mapping = {}
    for x in range(len(all_constraints)):
        mapping[all_constraints[x]] = x
    result = list(np.zeros(len(all_constraints), dtype=int))
    result[mapping[constraint]] = 1
    return result

