import numpy as np
from typing import List, Tuple
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class ImandraOptimizer:
    def __init__(self):
        self.gp = GaussianProcessRegressor(kernel=RBF(), random_state=20230805)
        self.y_samples = []

    def expected_improvement(self, x: List[float], y_samples: List[float], gp: GaussianProcessRegressor) -> float:
        mu, sigma = gp.predict([x], return_std=True)
        max_y = max(y_samples) if y_samples else 0.0
        z = (mu[0] - max_y) / sigma[0] if sigma[0] != 0.0 else 0.0
        ei = (mu[0] - max_y) * norm.cdf(z) + sigma[0] * norm.pdf(z)
        return -ei if sigma[0] != 0.0 else 0.0

    def minimize_ei(self, x0: List[float], bounds: List[Tuple[float, float]]) -> List[float]:
        from scipy.optimize import minimize
        result = minimize(
            lambda x: self.expected_improvement(x, self.y_samples, self.gp),
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        return result.x.tolist()

    def update_samples(self, y: float):
        self.y_samples.append(y)
        if len(self.y_samples) > 100:
            self.y_samples.pop(0)

if __name__ == "__main__":
    optimizer = ImandraOptimizer()
    x0 = [0.5, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    result = optimizer.minimize_ei(x0, bounds)
    optimizer.update_samples(0.123)
