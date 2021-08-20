#!/usr/bin/env python3
"""
0x03. Hyperparameter Tuning
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization on a noiseless 1D Gaussian process"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """init of BO"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = (np.linspace(start=bounds[0], stop=bounds[1],
                                num=ac_samples)[:, np.newaxis])
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ calculates the next best sample location """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            mu_sample_opt = np.min(self.gp.Y)
        else:
            mu_sample_opt = np.max(self.gp.Y)

        with np.errstate(divide="warn"):
            imp = mu_sample_opt - mu - self.xsi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei[ei == 0.0] = 0.0
        X = self.X_s[np.argmax(ei, axis=0)]
        return X, ei
