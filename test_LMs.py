from smoLM import *
import math
import numpy as np

A = LM(tokens, max_depth, seed=123456)
B = LM(tokens, max_depth, seed=654321)

for _ in range(5):
    print(to_string(A.sample_string()))
print()
for _ in range(5):
    print(to_string(B.sample_string()))

def MC_KL(A, B, n_reps=100000):
    print("Doing Monte Carlo for", n_reps)
    tot = 0
    for i in range(n_reps):
        sample = A.sample_string()
        p = A.prob(sample)
        q = B.prob(sample)
        if q == 0:
            tot += float('inf')
        else:
            tot += math.log2(p / q)

        print("so far:", tot / (i + 1), "\t\t", end='\r')

    print("Final estimation:", tot / n_reps)

KL_AB = calc_KL(A, B)
print("KL Divergence between A and B is", KL_AB)

MC_KL(A, B, 100)

Var_KL_AB = calc_MC_var(A, B, KL_AB)
print("The Variance of Monte Carlo is", Var_KL_AB)


# ============================================================
# KEY FUNCTION: Sample from proposal
# ============================================================
def sample_with_proposal(p_model, q_model, alpha=0.5):
    """
    Sample from proposal: r(x) = p(x) abs (log(p/q))*alpha + p(x)*(1-alpha)
    """
    state = []
    weight = 1.0

    while len(state) < p_model.depth and (len(state) == 0 or state[-1] != EOF):
        p_probs = p_model.layers[len(state)][*state]
        q_probs = q_model.layers[len(state)][*state]

        r_probs = p_probs * np.abs(np.log2((p_probs + 1e-12) / (q_probs + 1e-12)))*alpha + (1- alpha) * p_probs 
        r_probs = r_probs / np.sum(r_probs)

        next_token = np.random.choice(len(tokens), p=r_probs)
        weight *= p_probs[next_token] / r_probs[next_token]

        state.append(next_token)

    if state[-1] != EOF:
        state.append(EOF)

    return state, weight


# ============================================================
# Importance Sampling KL Estimator
# ============================================================
def IS_KL(A, B, n_reps=100, alpha=0.5):
    print("Importance Sampling with α =", alpha, ", n =", n_reps)
    tot = 0

    for i in range(n_reps):
        sample, weight = sample_with_proposal(A, B, alpha)

        p = A.prob(sample)
        q = B.prob(sample)

        if q == 0:
            tot += float('inf')
        else:
            tot += weight * math.log2(p / q)

        print(
            "Sample", i + 1, ":",
            round(tot / (i + 1), 4),
            end='\r'
        )

    print("\nFinal:", tot / n_reps)
    return tot / n_reps


# ============================================================
# Run Comparison
# ============================================================
print("=" * 60)
print("Computing TRUE KL divergence...")
KL_AB = calc_KL(A, B)
print("TRUE KL(A||B) =", round(KL_AB, 6))
print("=" * 60)

print("\nNaive MC (sample from A):")
MC_KL(A, B, 100)

print("\n" + "=" * 60)
is_est = IS_KL(A, B, 100, alpha=0.5)
print("=" * 60)

print("\nTrue value:", round(KL_AB, 6))
print("IS estimate:", round(is_est, 6))
print("Error:", round(abs(is_est - KL_AB), 6))


