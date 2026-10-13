import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    # 防止z过小 exp （-z）数值爆炸
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    X = np.asarray(X,float)
    y = np.asarray(y,float)
    N,D = X.shape
    w = np.zeros(D)
    b=0.0
    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        grad_w = X.T@(p-y)/N
        grad_b = np.mean(p-y)
        w-=lr*grad_w
        b-=lr*grad_b

    return w, b