import itertools
import numpy as np

def is_valid_mixed_strategy(A, support, prob):
    """
    Check if the mixed strategy is a Nash equilibrium with the given support.
    """
    n = A.shape[0]
    payoff = A @ prob
    max_payoff = max(payoff)

    for i in range(n):
        if i in support and not np.isclose(payoff[i], max_payoff):
            return False
        if i not in support and payoff[i] > max_payoff:
            return False

    return True

def brute_force_nash_equilibrium(A):
    """
    Find symmetric mixed Nash equilibrium by brute force.
    """
    n = A.shape[0]
    for k in range(1, n + 1):  # Size of the support
        for support in itertools.combinations(range(n), k):
            # Solve linear equations to find probabilities
            sub_matrix = A[np.ix_(support, support)]
            ones = np.ones(len(support))
            try:
                prob = np.linalg.solve(sub_matrix, ones)
                if all(prob > 0) and np.isclose(sum(prob), 1):
                    full_prob = np.zeros(n)
                    for i, s in enumerate(support):
                        full_prob[s] = prob[i]

                    if is_valid_mixed_strategy(A, support, full_prob):
                        return full_prob
            except np.linalg.LinAlgError:
                pass

    return None
