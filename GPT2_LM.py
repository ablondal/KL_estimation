

import math
import numpy as np
import matplotlib.pyplot as plt
from visualize_proposals import create_comprehensive_report
# from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def create_model_pair_llm(context="The movie was", device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create P and Q distributions from real LLMs.
    Args:
        context: input text prompt
    Returns:
        P_probs, Q_probs: numpy arrays of next-token probabilities
    """
    print(f"Loading models on {device}...")
    
    # Load base model (general)
    tokenizer_P = GPT2Tokenizer.from_pretrained('gpt2')
    model_P = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model_P.eval()
    
    # Load fine-tuned model (positive reviews)
    tokenizer_Q = GPT2Tokenizer.from_pretrained('lvwerra/gpt2-imdb')
    model_Q = GPT2LMHeadModel.from_pretrained('lvwerra/gpt2-imdb').to(device)
    model_Q.eval()
    
    # Tokenize context
    input_ids = tokenizer_P.encode(context, return_tensors='pt').to(device)
    
    # Get next-token probability distributions
    with torch.no_grad():
        # Model P (base)
        outputs_P = model_P(input_ids)
        logits_P = outputs_P.logits[0, -1, :]  # last token's logits
        P_probs = torch.softmax(logits_P, dim=0).cpu().numpy()
        
        # Model Q (fine-tuned)
        outputs_Q = model_Q(input_ids)
        logits_Q = outputs_Q.logits[0, -1, :]
        Q_probs = torch.softmax(logits_Q, dim=0).cpu().numpy()
    
    print(f"Vocabulary size: {len(P_probs)}")
    print(f"Context: '{context}'")
    
    # Show top tokens from each model
    top_k = 10
    top_P_indices = np.argsort(P_probs)[-top_k:][::-1]
    top_Q_indices = np.argsort(Q_probs)[-top_k:][::-1]
    
    print(f"\nTop {top_k} tokens from base model (P):")
    for idx in top_P_indices:
        token = tokenizer_P.decode([idx])
        print(f"  '{token}': {P_probs[idx]:.4f}")
    
    print(f"\nTop {top_k} tokens from positive model (Q):")
    for idx in top_Q_indices:
        token = tokenizer_Q.decode([idx])
        print(f"  '{token}': {Q_probs[idx]:.4f}")
    
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
        q = Q_probs[token]
        
        if q < eps:
            estimates.append(float('inf'))
        else:
            estimates.append(np.log(p / q))
    
    # Statistics
    estimate = np.mean(estimates)
    variance = np.var(estimates, ddof=1)
    std_error = np.sqrt(variance / n_samples)
    
    print(f"Final estimate: {estimate:.6f}")
    print(f"Empirical variance: {variance:.6f}")
    print(f"Standard error: {std_error:.6f}")
    
    return estimate, variance, std_error, estimates


def compute_proposal_chi2_aware(P_probs, Q_probs, alpha=0.5, eps=1e-12):
    """
    Proposal based on chi-square divergence: r ∝ P * sqrt(P/Q)  
    This corresponds to the χ²-divergence optimal proposal.
    """
    P_safe = np.clip(P_probs, eps, 1.0)
    Q_safe = np.clip(Q_probs, eps, 1.0)

    log_P = np.log(P_safe)
    log_Q = np.log(Q_safe)
    
    log_ratio = log_P - log_Q
    log_ratio = np.clip(log_ratio, -50, 50)
    log_r = log_P + alpha * log_ratio

    log_r_max = np.max(log_r)
    log_r_normalized = log_r - log_r_max
    r = np.exp(log_r_normalized)

    r = r / np.sum(r)    
    return r
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


def run_comparison_experiment(vocab_size=1000, n_samples=100, 
                              alpha_values=[0.3, 0.5, 0.7, 0.9],
                              divergence='medium'):
    """
    Run complete comparison of naive MC vs importance sampling.
    """
    print("=" * 80)
    print("KL DIVERGENCE ESTIMATION WITH LONG-TAILED DISTRIBUTIONS")
    print("=" * 80)
    
    # Create models P and Q
    P_probs, Q_probs = create_model_pair_llm(context="The movie was")
    vocab_size = len(P_probs)  # Update vocab_size from actual model    
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

    # Run experiment with real LLMs
    results, P_probs, Q_probs, r_probs_list = run_comparison_experiment(
        vocab_size=None,  # Will be determined by model
        n_samples=1000,   # Increase for real LLMs
        alpha_values=[0.3, 0.5, 0.7, 0.9],
        divergence=None   # Not used for LLMs
    )
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    ess_results = create_comprehensive_report(P_probs, Q_probs, r_probs_list)
