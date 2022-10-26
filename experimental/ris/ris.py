import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Integration/sampling domain
upper = 20

X = np.linspace(0, upper, 1000)

# Objective function
def f(x):
    return x**3 + x**2 + x

# Ground truth (numerical) integral
N_gt = 10**3
def gt_integral(x):
    X = np.linspace(0, x, N_gt)
    return np.sum(f(X)) * x/len(X)

# Noisy (MC) approximation
N = 10**2
def monte_carlo(x):
    r = np.random.uniform(0, x, N)
    return x/len(r) * np.sum(f(r))

# MC with importance sampling
def sample(x):
    # Sample higher at higher values
    eta = np.random.uniform(0, 1, N)

    samples = x * np.power(eta, 1/3)
    pdfs = 3 * samples**2/x**3

    return samples, pdfs

def importance_sampling(x):
    X, pdf = sample(x)
    return np.sum(f(X)/pdf)/len(X)

# MC with RIS
M = 10
def ris_sample(x):
    # Sample higher at higher values
    eta = np.random.uniform(0, 1, size=(N, M))
    
    # Get samples and pdfs, like before
    samples = x * np.power(eta, 1/3)
    pdfs = 3 * samples**2/x**3

    # Compute RIS weights using f as the target
    targets = f(samples)
    weights = targets/(pdfs * M)

    # Sample from each set of samples
    ris_samples = []
    ris_weights = []

    for i in range(N):
        wsum = np.sum(weights[i])
        probs = weights[i]/wsum

        sample = np.random.choice(samples[i], p=probs)
        ris_samples.append(sample)
        ris_weights.append(wsum/f(sample))

    ris_samples = np.array(ris_samples)
    ris_weights = np.array(ris_weights)

    return ris_samples, ris_weights

def ris(x):
    X, weights = ris_sample(x)
    return np.sum(f(X) * weights)/len(X)

# Use seaborn style
sns.set()

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(25, 15)

# Plot objective function
# ax1.plot(X, f(X), label='f(x)')

# Plot the integrals
integral_gt = np.vectorize(gt_integral)
integral_mc = np.vectorize(monte_carlo)
integral_is = np.vectorize(importance_sampling)
integral_ris = np.vectorize(ris)

R = np.linspace(upper * 0.8, upper, 1000)
ax1.plot(R, integral_gt(R), label='Ground truth')
ax1.plot(R, integral_mc(R), label='MC approximation')
ax1.plot(R, integral_is(R), label='MC with IS')
ax1.plot(R, integral_ris(R), label='MC with RIS')

# Distrbutions
S = sample(upper)
ax2.plot(S[0], S[1], 'o', label='Sampled points')

pdf_ftn = lambda x: 3 * x**2/upper**3
integral_pdf = np.sum(pdf_ftn(X)) * upper/len(X)

ax2.plot(X, pdf_ftn(X), label='true pdf(x)')
ax2.plot(X, integral_pdf * np.ones(len(X)), label='Integral of pdf')

# Show plot
ax1.legend()
ax2.legend()
plt.show()

# ris_sample(10)
