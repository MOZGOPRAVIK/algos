import numpy as np


def hungarian_algorithm(cost_matrix):

    n = len(cost_matrix)

    for row in cost_matrix:
        min_val = min(row)
        row -= min_val

    for col in range(n):
        min_val = min(cost_matrix[:, col])
        cost_matrix[:, col] -= min_val

    while True:
        zeros = (cost_matrix == 0)
        row_covered = np.zeros(n, dtype=bool)
        col_covered = np.zeros(n, dtype=bool)
        assignments = np.full(n, -1)

        for i in range(n):
            for j in range(n):
                if zeros[i, j] and not row_covered[i] and not col_covered[j]:
                    assignments[i] = j
                    row_covered[i] = True
                    col_covered[j] = True
                    break

        if all(assignments != -1):
            total_cost = sum(cost_matrix[i, assignments[i]] for i in range(n))
            return assignments, total_cost

        marked_rows = np.where(assignments == -1)[0]
        marked_cols = []
        new_marked_cols = []

        for row in marked_rows:
            for col in range(n):
                if zeros[row, col] and col not in marked_cols:
                    new_marked_cols.append(col)

        marked_cols = list(set(marked_cols + new_marked_cols))

        uncovered = np.ones_like(cost_matrix, dtype=bool)
        for row in range(n):
            if row not in marked_rows:
                uncovered[row, :] = False
        for col in marked_cols:
            uncovered[:, col] = False

        min_val = np.min(cost_matrix[uncovered])

        for row in range(n):
            for col in range(n):
                if uncovered[row, col]:
                    cost_matrix[row, col] -= min_val
                elif row not in marked_rows and col in marked_cols:
                    cost_matrix[row, col] += min_val
