import numpy as np

epsilon = 1e-3
lambda_max = 1e-3

def DLS(J):
    U, s, V = np.linalg.svd(J, full_matrices = False)
    s_min = np.min(s)
    if s_min > epsilon:
        lambda_squ = 0
    else:
        lambda_squ = (1 - (s_min / epsilon) ** 2) * lambda_max ** 2
    J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_squ * np.eye(J.shape[0]))

    return J_pinv
     