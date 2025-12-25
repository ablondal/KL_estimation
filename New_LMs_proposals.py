"""
KL Divergence Estimation with Long-Tailed Distributions
Compares Naive MC vs Importance Sampling on realistic language model distributions
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zipf
from visualize_proposals import create_comprehensive_report

np.random.seed(62)

class LongTailedDistribution:
    
    @staticmethod
    def zipfian(vocab_size, alpha=1.5):
        """
        Zipfian distribution: P(k) ∝ 1/k^alpha
        Args:
        vocab_size: number of tokens in vocabulary
        alpha: controls heaviness of tail (1.0-1.5 typical for language)
        """
        ranks = np.arange(1, vocab_size + 1)
        probs = 1.0 / (ranks ** alpha)
        probs = probs / probs.sum()
        return probs

    @staticmethod
    def mixture_distribution(vocab_size, components=3): # set vocab to >1000
        """
        Real LLM distributions often look like mixtures:
        - Very heavy head (top ~100 tokens)
        - Power law middle
        - Very long flat tail
        """
        probs = np.zeros(vocab_size)
        
        # Component 1: Very heavy head (top 100)
        head_size = min(100, vocab_size // 10)
        probs[:head_size] = np.exp(-0.05 * np.arange(head_size))
        
        # Component 2: Power law middle
        middle_start = head_size
        middle_end = vocab_size // 2
        middle_indices = np.arange(middle_start, middle_end)
        probs[middle_indices] = 1.0 / (middle_indices ** 1.3)
        
        # Component 3: Uniform-ish tail
        tail_start = middle_end
        tail_indices = np.arange(tail_start, vocab_size)
        probs[tail_indices] = 1.0 / (tail_indices ** 0.7)
        
        # Mix with some randomness
        probs *= np.random.lognormal(0, 0.5, vocab_size)
        
        probs = probs / probs.sum()
        return probs
    
    @staticmethod
    def zipfian_with_cutoff(vocab_size, alpha=1.5, cutoff_rank=100):
        """
        Real LLMs often have power law head + exponential tail
        """
        ranks = np.arange(1, vocab_size + 1)
        probs = 1.0 / (ranks ** alpha)
        
        # Apply exponential cutoff after certain rank
        cutoff_mask = ranks > cutoff_rank
        decay_factor = np.exp(-(ranks[cutoff_mask] - cutoff_rank) / (vocab_size/10))
        probs[cutoff_mask] *= decay_factor
        
        probs = probs / probs.sum()
        return probs
        
    @staticmethod
    def perturb_distribution(base_probs, shift_ratio=0.2, n_shift=100):
        """
        Create a similar but different distribution by shifting probability mass.
        """
        vocab_size = len(base_probs)
        perturbed = base_probs.copy()
        
        # Randomly select tokens to boost/reduce
        boost_indices = np.random.choice(
            range(n_shift, vocab_size),
            size=n_shift // 2,
            replace=False
        )
        
        reduce_indices = np.random.choice(
            range(n_shift, vocab_size // 2),
            size=n_shift // 2,
            replace=False
        )
        
        # Shift probability mass
        shift_mass = base_probs.sum() * shift_ratio
        mass_per_boost = shift_mass / len(boost_indices)
        mass_per_reduce = shift_mass / len(reduce_indices)
        
        perturbed[boost_indices] += mass_per_boost
        perturbed[reduce_indices] = np.maximum(
            perturbed[reduce_indices] - mass_per_reduce,
            1e-10
        )
        
        # Renormalize
        perturbed = perturbed / perturbed.sum()
        return perturbed


def create_model_pair(vocab_size=1000, divergence='medium'):
    """
    Create two related long-tailed distributions P and Q.
    Args:
        vocab_size: size of vocabulary
        divergence: 'low', 'medium', or 'high' - controls how different P and Q are
    """
    # Base distribution P (base model)
    P_probs = LongTailedDistribution.zipfian(vocab_size, alpha=1.5)
    
    # Perturbed distribution Q (fine-tuned model)
    if divergence == 'low':
        Q_probs = LongTailedDistribution.perturb_distribution(P_probs, shift_ratio=0.1)
    elif divergence == 'medium':
        Q_probs = LongTailedDistribution.perturb_distribution(P_probs, shift_ratio=0.3)
    elif divergence == 'high':
        Q_probs = LongTailedDistribution.zipfian(vocab_size, alpha=1.2)
        Q_probs = LongTailedDistribution.perturb_distribution(Q_probs, shift_ratio=0.5)
    else:
        raise ValueError("divergence must be 'low', 'medium', or 'high'")
    
    return P_probs, Q_probs

def create_model_pair_zipCutoff(vocab_size=2000, divergence='medium'):
    P_probs = LongTailedDistribution.zipfian_with_cutoff(vocab_size)
    
    if divergence == 'low':
        Q_probs = LongTailedDistribution.perturb_distribution(P_probs, shift_ratio=0.1)
    elif divergence == 'medium':
        Q_probs = LongTailedDistribution.perturb_distribution(P_probs, shift_ratio=0.3)
    elif divergence == 'high':
        Q_probs = LongTailedDistribution.zipfian_with_cutoff(vocab_size, alpha=1.2)
        Q_probs = LongTailedDistribution.perturb_distribution(Q_probs, shift_ratio=0.5)
    else:
        raise ValueError("divergence must be 'low', 'medium', or 'high'")
    
    return P_probs, Q_probs


def create_model_pair_mixture(vocab_size=2000, divergence='medium'):
    P_probs = LongTailedDistribution.mixture_distribution(vocab_size)
    
    if divergence == 'low':
        Q_probs = LongTailedDistribution.perturb_distribution(P_probs, shift_ratio=0.1)
    elif divergence == 'medium':
        Q_probs = LongTailedDistribution.perturb_distribution(P_probs, shift_ratio=0.3)
    elif divergence == 'high':
        Q_probs = LongTailedDistribution.zipfian_with_cutoff(vocab_size, alpha=1.2)
        Q_probs = LongTailedDistribution.perturb_distribution(Q_probs, shift_ratio=0.5)
    else:
        raise ValueError("divergence must be 'low', 'medium', or 'high'")
    
    return P_probs, Q_probs


def compute_true_KL(P_probs, Q_probs, eps=1e-12):
    P_safe = np.clip(P_probs, eps, 1.0)
    Q_safe = np.clip(Q_probs, eps, 1.0)
    
    kl = np.sum(P_safe * np.log(P_safe / Q_safe))
    return kl


def compute_theoretical_variance(P_probs, Q_probs, true_kl, eps=1e-12):
    P_safe = np.clip(P_probs, eps, 1.0)
    Q_safe = np.clip(Q_probs, eps, 1.0)
    
    log_ratios = np.log(P_safe / Q_safe)
    
    # Var[log(P/Q)] = E[(log(P/Q))²] - (E[log(P/Q)])²
    second_moment = np.sum(P_safe * log_ratios**2)
    variance = second_moment - true_kl**2
    
    return variance

def sample_from_distribution(probs, n_samples=1):
    vocab_size = len(probs)
    samples = np.random.choice(vocab_size, size=n_samples, p=probs)
    return samples

def compute_proposal_balanced2 (P_probs, Q_probs, eps=1e-12):
    # r = np.sqrt(P_safe * Q_safe * (log_ratio**2 + 1))

    P_safe = np.clip(P_probs, eps, 1.0)
    Q_safe = np.clip(Q_probs, eps, 1.0)
    
    log_ratio = np.log(P_safe / Q_safe)
    r_probs = np.sqrt(P_safe * Q_safe * (log_ratio**2 + 1))
    
    return r_probs / r_probs.sum()    

def compute_proposal_balanced1 (P_probs, Q_probs, eps=1e-12):
    # r = np.sqrt(P_safe * Q_safe * (log_ratio**2 + 1))

    P_safe = np.clip(P_probs, eps, 1.0)
    Q_safe = np.clip(Q_probs, eps, 1.0)
    
    log_ratio = np.log(P_safe / Q_safe)
    r_probs = np.sqrt(P_safe * Q_safe) * np.abs(log_ratio)
    
    return r_probs / r_probs.sum()  

def compute_proposal_balanced(P_probs, Q_probs, eps=1e-12):
    """
    Balanced proposal: r ∝ sqrt(P * |log(P/Q)| * Q / (P + Q))
    Tries to balance P's coverage with high-divergence regions
    """
    P_safe = np.clip(P_probs, eps, 1.0)
    Q_safe = np.clip(Q_probs, eps, 1.0)
    
    log_ratio = np.log(P_safe / Q_safe)
    r_probs = np.sqrt(P_safe * np.abs(log_ratio) * Q_safe / (P_safe + Q_safe + eps))
    
    return r_probs / r_probs.sum()

def compute_proposal_adaptive_mixture(P_probs, Q_probs, eps=1e-12):
    """
    Adaptive: more weight on high-dKL when differences are large
    r ∝ P * (1 + |log(P/Q)|)
    """
    P_safe = np.clip(P_probs, eps, 1.0)
    Q_safe = np.clip(Q_probs, eps, 1.0)
    
    log_ratio = np.log(P_safe / Q_safe)
    r_probs = P_safe * (1 + np.abs(log_ratio))
    return r_probs / r_probs.sum()
    
def compute_proposal_alpha_mixture(P_probs, Q_probs, alpha=0.5, eps=1e-12):
    """
    Compute proposal distribution:
    r(x) = P(x) * |log(P/Q)| * alpha + P(x) * (1-alpha)
    """
    P_safe = np.clip(P_probs, eps, 1.0)
    Q_safe = np.clip(Q_probs, eps, 1.0)
    
    log_ratio = np.log(P_safe / Q_safe)
    abs_log_ratio = np.abs(log_ratio)
    
    # r(x) = P(x) * |log(P/Q)| * alpha + P(x) * (1-alpha)
    r_probs = P_safe * abs_log_ratio * alpha + P_safe * (1 - alpha)
    
    # Normalize
    r_probs = r_probs / np.sum(r_probs)
    
    return r_probs


def compute_proposal_optimal(P_probs, Q_probs, eps=1e-12):
    """
    Theoretically optimal: r ∝ P|log(P/Q)|
    Serves as a performance ceiling
    Intractable for LLMs
    """
    P_safe = np.clip(P_probs, eps, 1.0)
    Q_safe = np.clip(Q_probs, eps, 1.0)
    r_probs = P_safe * np.abs(np.log(P_safe / Q_safe))
    return r_probs / r_probs.sum()


def compute_proposal_geometric_mixture(P_probs, Q_probs, beta=0.5, eps=1e-12):
    """
    geometric mixture: r ∝ P^beta * Q^(1-beta)
    geometric mean with beta=0.5
    """
    P_safe = np.clip(P_probs, eps, 1.0)
    Q_safe = np.clip(Q_probs, eps, 1.0)
    r_probs = (P_safe ** beta) * (Q_safe ** (1 - beta))
    return r_probs / r_probs.sum()


def compute_proposal_mixture(P_probs, Q_probs, lambda_mix=0.5, eps=1e-12):
    """
    r = λP + (1-λ)Q - simple mixture
    doesn't explicitly target high divergence regions
    """
    return lambda_mix * P_probs + (1 - lambda_mix) * Q_probs


def naive_MC_estimator(P_probs, Q_probs, n_samples=100, eps=1e-12):
    print(f"Naive Monte Carlo with {n_samples} samples")
    
    # Sample from P
    samples = sample_from_distribution(P_probs, n_samples)
    
    # Compute log(P/Q) for each sample
    estimates = []
    for token in samples:
        p = P_probs[token]
        q = max(Q_probs[token], eps)
        estimates.append(np.log(p / q))
    
    estimate = np.mean(estimates)
    variance = np.var(estimates, ddof=1)
    std_error = np.sqrt(variance / n_samples)
    
    print(f"Final estimate: {estimate:.6f}")
    print(f"Empirical variance: {variance:.6f}")
    print(f"Standard error: {std_error:.6f}")
    
    return estimate, variance, std_error, estimates


def importance_sampling_estimator(P_probs, Q_probs, n_samples=100, alpha_values=[0.3, 0.5, 0.7, 0.9], eps=1e-12):
    print(f"Importance Sampling with n={n_samples}")

    r_probs_list = []

    for alpha in alpha_values:
        r_probs_list.append({
            'name': f'Alpha-mixture (α={alpha})', 
            'r': compute_proposal_alpha_mixture(P_probs, Q_probs, alpha=alpha, eps=eps)
        })
        r_probs_list.append({
            'name': f'Geometric (α={alpha})', 
            'r': compute_proposal_geometric_mixture(P_probs, Q_probs, beta=alpha, eps=eps)
        })
        r_probs_list.append({
            'name': f'Mixture (α={alpha})', 
            'r': compute_proposal_mixture(P_probs, Q_probs, lambda_mix=alpha, eps=eps)
        })
    
    r_probs_list.extend([
        {'name': 'Balanced', 'r': compute_proposal_balanced(P_probs, Q_probs, eps)}
    ])    
    r_probs_list.extend([
        {'name': 'Balanced1', 'r': compute_proposal_balanced1(P_probs, Q_probs, eps)}
    ])    
    r_probs_list.extend([
        {'name': 'Balanced2', 'r': compute_proposal_balanced2(P_probs, Q_probs, eps)}
    ])
    r_probs_list.extend([
        {'name': 'Adaptive', 'r': compute_proposal_adaptive_mixture(P_probs, Q_probs, eps)}
    ])
    r_probs_list.extend([
        {'name': 'Optimal', 'r': compute_proposal_optimal(P_probs, Q_probs, eps)}
    ])

    all_results = []

    for proposal in r_probs_list:
        print(f"\nTesting {proposal['name']}")
        # Sample from proposal
        samples = sample_from_distribution(proposal['r'], n_samples)
        
        # Compute weighted log(P/Q) for each sample
        estimates = []
        for token in samples:
            p = P_probs[token]
            q = Q_probs[token]
            r = proposal['r'][token]
            
            # Importance weight: w = P(x) / r(x)
            weight = p / r
            
            if q < eps:
                estimates.append(float('inf'))
            else:
                estimates.append(weight * np.log(p / q))
        
        # Statistics
        estimate = np.mean(estimates)
        variance = np.var(estimates, ddof=1)
        std_error = np.sqrt(variance / n_samples)
        
        print(f"Final estimate: {estimate:.6f}")
        print(f"Empirical variance: {variance:.6f}")
        print(f"Standard error: {std_error:.6f}")

        all_results.append({
            'name': proposal['name'],
            'estimate': estimate,
            'variance': variance,
            'std_error': std_error,
            'estimates': estimates
        })
    
    return r_probs_list, all_results


def run_comparison_experiment(vocab_size=1000, model_type="create_model_pair", n_samples=100, 
                              alpha_values=[0.3, 0.5, 0.7, 0.9],
                              divergence='medium'):
    """
    Run complete comparison of naive MC vs importance sampling.
    """
    print("=" * 80)
    print("KL DIVERGENCE ESTIMATION WITH LONG-TAILED DISTRIBUTIONS")
    print("=" * 80)
    
    # Create models P and Q
    print(f"\nCreating long-tailed distributions (vocab_size={vocab_size}) with {model_type}...")
    if model_type == "create_model_pair":
        P_probs, Q_probs = create_model_pair(vocab_size, divergence=divergence)
    elif model_type == "create_model_pair_zipCutoff":
        P_probs, Q_probs = create_model_pair_zipCutoff(vocab_size, divergence=divergence)
    elif model_type == "create_model_pair_mixture":
        P_probs, Q_probs = create_model_pair_mixture(vocab_size, divergence=divergence)
    else:
        raise ValueError("model_type must be create_model_pair, create_model_pair_zipCutoff, or create_model_pair_mixture")
        
    # Show distribution statistics
    print(f"\nDistribution statistics:")
    print(f"  Top 10 tokens in P: {P_probs[:10].sum():.2%}")
    print(f"  Top 100 tokens in P: {P_probs[:100].sum():.2%}")
    print(f"  Top 10 tokens in Q: {Q_probs[:10].sum():.2%}")
    print(f"  Top 100 tokens in Q: {Q_probs[:100].sum():.2%}")
    
    # Compute true KL
    print("\n" + "=" * 80)
    print("Computing TRUE KL divergence...")
    true_kl = compute_true_KL(P_probs, Q_probs)
    theoretical_variance = compute_theoretical_variance(P_probs, Q_probs, true_kl)
    
    print(f"TRUE KL(P||Q) = {true_kl:.6f}")
    print(f"Theoretical MC variance = {theoretical_variance:.6f}")
    print("=" * 80)
    
    results = {
        'true_kl': true_kl,
        'theoretical_variance': theoretical_variance,
        'methods': []
    }
    
    # Naive Monte Carlo
    print("\n[METHOD 1] Naive Monte Carlo")
    print("-" * 80)
    mc_est, mc_var, mc_se, mc_estimates = naive_MC_estimator(
        P_probs, Q_probs, n_samples
    )
    
    results['methods'].append({
        'name': 'Naive MC',
        'estimate': mc_est,
        'variance': mc_var,
        'std_error': mc_se,
        'error': abs(mc_est - true_kl),
        'estimates': mc_estimates
    })

    # various types of importance sampling
    print("\n" + "=" * 80)
    print(f"[METHOD 2] Importance Sampling")
    r_probs_list, all_results = importance_sampling_estimator(P_probs, Q_probs, n_samples, alpha_values)
    results['methods'].extend(all_results)
    
    return results, P_probs, Q_probs, r_probs_list


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Run experiment with long-tailed distributions
    for model_type in ["create_model_pair", "create_model_pair_zipCutoff", "create_model_pair_mixture"]:
        results, P_probs, Q_probs, r_probs_list = run_comparison_experiment(
            vocab_size=3000,
            model_type=model_type,
            n_samples=300,
            alpha_values=[0.3, 0.5, 0.7, 0.9],
            divergence='medium'
        )
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        ess_results = create_comprehensive_report(P_probs, Q_probs, model_type, r_probs_list)
