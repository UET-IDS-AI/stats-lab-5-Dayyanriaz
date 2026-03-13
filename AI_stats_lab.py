import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.

    P(a < X < b) = e^(-lam*a) - e^(-lam*b)
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    count = np.sum((samples > a) & (samples < b))
    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    coef = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coef * exponent


def posterior_probability(time):
    """
    Compute P(B | X = time) using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """

    muA, sigmaA = 40, 2
    muB, sigmaB = 45, 2

    pA = 0.3
    pB = 0.7

    likelihood_A = gaussian_pdf(time, muA, sigmaA)
    likelihood_B = gaussian_pdf(time, muB, sigmaB)

    numerator = likelihood_B * pB
    denominator = likelihood_A * pA + likelihood_B * pB

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    muA, sigmaA = 40, 2
    muB, sigmaB = 45, 2

    pA = 0.3
    pB = 0.7

    # simulate group labels
    groups = np.random.choice(["A", "B"], size=n, p=[pA, pB])

    times = np.zeros(n)

    # generate times
    times[groups == "A"] = np.random.normal(muA, sigmaA, np.sum(groups == "A"))
    times[groups == "B"] = np.random.normal(muB, sigmaB, np.sum(groups == "B"))

    # approximate conditioning around given time
    tol = 0.1
    mask = np.abs(times - time) < tol

    if np.sum(mask) == 0:
        return 0

    return np.sum(groups[mask] == "B") / np.sum(mask)
