import numpy as np
import torch
import math
import json
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any, Tuple

PROMPT_CATEGORIES = {
    'factual': [
        "An interesting fact is that",
        "The scientific discovery shows that",
        "Historically, it was known that"
    ],
    'creative': [
        "Once upon a time, there was",
        "In a distant galaxy, the",
        "The magical forest contained"
    ],
    'instructional': [
        "To solve this problem, first",
        "The best way to approach this is",
        "Following these steps will"
    ],
    'reasoning': [
        "Given that X is true, then",
        "If we assume Y, it follows that",
        "The logical conclusion is that"
    ]
}

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

model_a_id = "Qwen/Qwen2.5-1.5B"
model_b_id = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_a_id)

model_a = AutoModelForCausalLM.from_pretrained(
    model_a_id,
    torch_dtype=dtype
).to(device).eval()

model_b = AutoModelForCausalLM.from_pretrained(
    model_b_id,
    torch_dtype=dtype
).to(device).eval()

def kl_divergence_from_logits(logits_p, logits_q, eps=1e-8):
    """
    KL(P || Q) for a single timestep
    logits_* : [vocab]
    """
    log_p = torch.log_softmax(logits_p, dim=-1)
    log_q = torch.log_softmax(logits_q, dim=-1)

    p = torch.exp(log_p)
    kl = torch.sum(p * (log_p - log_q))
    return kl

def calc_kl_RB(
        prompt,
        max_new_tokens=50,
        temperature=1):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    kl_expectations = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            out_a = model_a(input_ids); out_b = model_b(input_ids)

            logits_a = out_a.logits[0, -1] / temperature
            logits_b = out_b.logits[0, -1] / temperature

            l_p = torch.log_softmax(logits_a, dim=-1)
            l_q = torch.log_softmax(logits_b, dim=-1)
            p = torch.exp(l_p)

            # --- calc kl ---
            kl_expectations.append(torch.sum(p * (l_p - l_q)).item())

            # --- sampling step (modifiable) ---
            next_token = torch.multinomial(p, num_samples=1)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0]), kl_expectations

def batch_sample_once(
    prompt,
    next_token_weight,
    max_new_tokens=50,
    temperature=1,
    batch_size=32
):
    """Batch version of sample_once with proper padding for variable-length sequences"""
    
    # Prepare initial inputs
    input_texts = [prompt] * batch_size
    encoding = tokenizer(input_texts, return_tensors="pt", padding=True)
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    
    # Store original lengths
    original_length = input_ids.shape[1]
    
    # Initialize logging tensors
    batch_log_r = torch.zeros(batch_size, device=device)
    batch_log_p = torch.zeros(batch_size, device=device)
    batch_log_q = torch.zeros(batch_size, device=device)
    
    # Track active sequences (not terminated by EOS)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    # Store all generated tokens in a list for each sequence
    # This is more flexible than trying to maintain a fixed tensor
    generated_tokens = [[] for _ in range(batch_size)]
    
    for step in range(max_new_tokens):
        if not active_mask.any():
            break
        
        # Get indices of active sequences
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        num_active = len(active_indices)
        
        # Prepare inputs for active sequences
        # We need to reconstruct the full sequence for each active sample
        active_sequences = []
        for idx in active_indices:
            # Get the original tokens plus generated tokens
            seq_tokens = input_ids[idx].tolist()
            seq_tokens = [t for t, mask in zip(seq_tokens, attention_mask[idx].tolist()) if mask == 1]
            seq_tokens.extend(generated_tokens[idx])
            active_sequences.append(seq_tokens)
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in active_sequences)
        padded_sequences = []
        padded_masks = []
        
        for seq in active_sequences:
            padded_seq = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
            padded_mask = [1] * len(seq) + [0] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
            padded_masks.append(padded_mask)
        
        # Convert to tensors
        active_inputs = torch.tensor(padded_sequences, device=device, dtype=torch.long)
        active_attention = torch.tensor(padded_masks, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # Get model outputs
            out_a = model_a(active_inputs, attention_mask=active_attention)
            out_b = model_b(active_inputs, attention_mask=active_attention)
            
            # Get logits for the last token
            logits_a = out_a.logits[:, -1] / temperature
            logits_b = out_b.logits[:, -1] / temperature
            
            # Compute probabilities
            l_p = torch.log_softmax(logits_a, dim=-1)
            l_q = torch.log_softmax(logits_b, dim=-1)
            p = torch.exp(l_p)
            
            # Apply proposal function to get sampling probabilities
            probs_list = []
            for i, idx in enumerate(active_indices):
                current_log_p = batch_log_p[idx].item()
                current_log_q = batch_log_q[idx].item()
                
                prob = next_token_weight(
                    p[i], l_p[i], l_q[i],
                    current_log_p, current_log_q
                )
                probs_list.append(prob)
            
            # Stack probabilities
            probs = torch.stack(probs_list)
            
            # Sample next tokens
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Update logs and store generated tokens
            for i, idx in enumerate(active_indices):
                token = next_tokens[i].item()
                generated_tokens[idx].append(token)
                
                # Get the probabilities for this specific token
                log_p_val = l_p[i, next_tokens[i]]
                log_q_val = l_q[i, next_tokens[i]]
                log_r_val = torch.log(probs[i, next_tokens[i]] + 1e-12)
                
                # Update cumulative logs
                batch_log_r[idx] += log_r_val
                batch_log_p[idx] += log_p_val
                batch_log_q[idx] += log_q_val
                
                # Check for EOS
                if token == tokenizer.eos_token_id:
                    active_mask[idx] = False
    
    # Construct final sequences and decode
    all_texts = []
    for i in range(batch_size):
        # Combine original tokens and generated tokens
        original_seq = input_ids[i].tolist()
        original_seq = [t for t, mask in zip(original_seq, attention_mask[i].tolist()) if mask == 1]
        full_seq = original_seq + generated_tokens[i]
        text = tokenizer.decode(full_seq, skip_special_tokens=True)
        all_texts.append(text)
    
    return all_texts, batch_log_r.cpu().numpy(), batch_log_p.cpu().numpy(), batch_log_q.cpu().numpy()
    
def sample_once(
        prompt,
        next_token_weight=(lambda p, l_p, l_q, lg_p, lg_q: p),
        max_new_tokens=50,
        temperature=1):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    log_r = 0
    log_p = 0     # sum_t log P(x_t | x_{x<t}); cumulative divergence
    log_q = 0

    p_probs = []
    q_probs = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            out_a = model_a(input_ids)
            out_b = model_b(input_ids)

            logits_a = out_a.logits[0, -1] / temperature
            logits_b = out_b.logits[0, -1] / temperature

            l_p = torch.log_softmax(logits_a, dim=-1) # log P(x_t|x_<t); current token divergence
            l_q = torch.log_softmax(logits_b, dim=-1)
            p = torch.exp(l_p)

            # --- sampling step (modifiable) ---
            probs = next_token_weight(p, l_p, l_q, log_p, log_q)
            next_token = torch.multinomial(probs, num_samples=1)

            # For numerical stability, we really want to use Python floats
            # as early as possible
            log_r += math.log(probs[next_token].item())
            log_p += l_p[next_token].item()
            log_q += l_q[next_token].item()

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0]), log_r, log_p, log_q

# ========== Proposal Functions ==========

def just_p(p, l_p, l_q, lg_p, lg_q):
    return p

def eps_01(p, l_p, l_q, lg_p, lg_q):
    prop = p*torch.abs(l_p - l_q + lg_p - lg_q)
    prop = 0.9*torch.nn.functional.normalize(prop, p=1, dim=0) + 0.1*p
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def eps_03(p, l_p, l_q, lg_p, lg_q):
    prop = p*torch.abs(l_p - l_q + lg_p - lg_q)
    prop = 0.7*torch.nn.functional.normalize(prop, p=1, dim=0) + 0.3*p
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def eps_05(p, l_p, l_q, lg_p, lg_q):
    prop = p*torch.abs(l_p - l_q + lg_p - lg_q)
    prop = 0.5*torch.nn.functional.normalize(prop, p=1, dim=0) + 0.5*p
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def eps_09(p, l_p, l_q, lg_p, lg_q):
    prop = p*torch.abs(l_p - l_q + lg_p - lg_q)
    prop = 0.1*torch.nn.functional.normalize(prop, p=1, dim=0) + 0.9*p
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def balanced(p, l_p, l_q, lg_p, lg_q):
    # prop to sqrt(P * |log(P/Q)| * Q / (P + Q))
    q = torch.exp(l_q)
    prop = torch.sqrt((p * q * torch.abs(l_p - l_q + lg_p - lg_q)) / (p + q + 1e-5))
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def exponential_family(p, l_p, l_q, lg_p, lg_q, beta=1.0):
    # r ∝ P * exp(beta * |log(P/Q)|)
    log_ratio = l_p - l_q + (lg_p - lg_q)
    prop = p * torch.exp(beta * torch.abs(log_ratio))
    return torch.nn.functional.normalize(prop, p=1, dim=0)
    
def cross_entropy(p, l_p, l_q, lg_p, lg_q, beta=1.0):
    # r ∝ P * exp(beta * (-P * log Q))
    q = torch.exp(l_q)
    ptwise_ce = -p * l_q
    prop = p * torch.exp(beta * ptwise_ce)
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def adaptive(p, l_p, l_q, lg_p, lg_q):
    # prop to P * (1 + |log(P/Q)|)
    prop = p * (1 + torch.abs(l_p - l_q + lg_p - lg_q))
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def mix09(p, l_p, l_q, lg_p, lg_q):
    # prop to r = λP + (1-λ)Q for lambda 0.9
    prop = 0.9 * p + 0.1 * torch.exp(l_q)
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def mix03(p, l_p, l_q, lg_p, lg_q):
    # prop to r = λP + (1-λ)Q for lambda 0.3
    prop = 0.3 * p + 0.7 * torch.exp(l_q)
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def balanced2(p, l_p, l_q, lg_p, lg_q):
    # prop to r = sqrt(P * Q * (log_ratio**2 + 1))
    q = torch.exp(l_q)
    prop = torch.sqrt(p * q * (torch.pow(torch.abs(l_p - l_q + lg_p - lg_q), 2) + 1))
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def geometric(p, l_p, l_q, lg_p, lg_q):
    q = torch.exp(l_q)
    # prop to P^beta * Q^(1-beta) with beta = 0.3
    prop = torch.pow(p, 0.3) * torch.pow(q, 0.7)
    return torch.nn.functional.normalize(prop, p=1, dim=0)

# ========== Helper Functions ==========

def compute_baseline_RB(prompt, n_runs=10, max_new_tokens=20, temperature=0.4):
    """
    Compute Rao-Blackwellized baseline KL estimate.
    Returns: (mean_kl, variance, all_kls)
    """
    kls = []  # Store total KL for each run
    kl_sum = 0
    
    print(f"Computing Rao-Blackwellized baseline (n={n_runs})...")
    
    for k in range(n_runs):
        text, kl_expectations = calc_kl_RB(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        total_kl = sum(kl_expectations)
        kls.append(total_kl)
        kl_sum += total_kl
        
        current_mean = kl_sum / (k + 1)
        print(f"Run {k+1}/{n_runs}: Current mean KL = {current_mean:.6f}", end='\r')
    
    print()  # New line after progress
    mean_kl = kl_sum / n_runs
    
    # Compute variance
    var_sum = sum((kl - mean_kl) ** 2 for kl in kls)
    variance = var_sum / n_runs if n_runs > 1 else 0

    print(f"Variance: {variance}")
    return mean_kl, variance, kls
    
def compute_variance(particles: List[Dict], kl_est: float, w_sum: float) -> float:
    """Compute variance of KL estimate"""
    if len(particles) == 0 or w_sum == 0: 
        return float('inf') 
                    
    N = len(particles)
    weighted_sqr_error_sum = 0.0

    for p in particles: 
        weighted_sqr_error_sum += (p['weight'] ** 2) * ((p['val'] - kl_est) ** 2)

    variance = (1.0/N) * (weighted_sqr_error_sum / (w_sum ** 2))
    return variance 

def compute_effective_sample_size(particles: List[Dict]) -> float:
    """Compute effective sample size"""
    if len(particles) == 0:
        return 0.0
    
    weights = np.array([p['weight'] for p in particles])
    w_sum = np.sum(weights)
    
    if w_sum == 0:
        return 0.0
    
    norm_weights = weights / w_sum
    ess = 1.0 / np.sum(norm_weights ** 2)
    return ess
    
def compute_kl_with_different_averaging(particles: List[Dict]) -> Dict:
    """Compare different weighting/averaging strategies"""
    
    weights = np.array([p['weight'] for p in particles])
    values = np.array([p['val'] for p in particles])
    
    if len(weights) == 0 or np.sum(weights) == 0:
        return {
            'standard': 0.0,
            'clipped': 0.0,
            'bayesian': 0.0,
            'bootstrap_ci': (0.0, 0.0),
            'bootstrap_mean': 0.0,
            'weight_entropy': 0.0
        }
    
    # 1. Standard importance sampling
    kl_standard = np.average(values, weights=weights)
    
    # 2. Self-normalized with clipping
    clipped_weights = np.clip(weights, np.percentile(weights, 5), 
                               np.percentile(weights, 95))
    kl_clipped = np.average(values, weights=clipped_weights)
    
    # 3. Bayesian averaging
    prior_strength = 0.1
    prior_mean = np.median(values)
    weighted_var = np.average((values - kl_standard)**2, weights=weights)
    kl_bayesian = (prior_strength * prior_mean + kl_standard) / (1 + prior_strength)
    
    # 4. Bootstrap confidence intervals
    bootstrap_estimates = []
    for _ in range(100):
        idx = np.random.choice(len(particles), size=len(particles), replace=True)
        bs_weights = weights[idx]
        bs_values = values[idx]
        if np.sum(bs_weights) > 0:
            bootstrap_estimates.append(np.average(bs_values, weights=bs_weights))
    
    ci_lower = np.percentile(bootstrap_estimates, 2.5) if bootstrap_estimates else 0.0
    ci_upper = np.percentile(bootstrap_estimates, 97.5) if bootstrap_estimates else 0.0
    
    # Weight entropy
    normalized_weights = weights / np.sum(weights)
    weight_entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-10))

    bootstrap_mean = np.mean(bootstrap_estimates) if bootstrap_estimates else 0.0

    print("\nKL with different averaging techniques:")
    print(f"\nStandard: {float(kl_standard)}")
    print(f"\nClipped: {float(kl_clipped)}")
    print(f"\nbayesian: {float(kl_bayesian)}")
    print(f"\nbootstrap_ci: {(float(ci_lower), float(ci_upper))}")
    print(f"\nbootstrap_mean: {float(bootstrap_mean)}")
    print(f"\nweight_entropy: {float(weight_entropy)}")
    
    return {
        'standard': float(kl_standard),
        'clipped': float(kl_clipped),
        'bayesian': float(kl_bayesian),
        'bootstrap_ci': (float(ci_lower), float(ci_upper)),
        'bootstrap_mean': float(bootstrap_mean),
        'weight_entropy': float(weight_entropy)
    }

def run_batch_experiment(
    proposal_func,
    proposal_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int = 20,
    n_reps: int = 200,
    batch_size: int = 32
) -> Dict:
    """Run experiment using batch processing - FIXED VERSION"""
    
    print(f"  Testing {proposal_name} at temp={temperature} with prompt: '{prompt[:30]}...'")
    
    particles = []
    n_batches = math.ceil(n_reps / batch_size)
    
    for batch_idx in range(n_batches):
        current_batch_size = min(batch_size, n_reps - batch_idx * batch_size)
        
        if current_batch_size <= 0:
            break
        
        try:
            texts, log_r, log_p, log_q = batch_sample_once(
                prompt=prompt,
                next_token_weight=proposal_func,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=current_batch_size
            )
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
        
        # Process results
        for i in range(len(texts)):
            # Ensure we have valid numbers
            if (np.isfinite(log_p[i]) and np.isfinite(log_r[i]) and 
                np.isfinite(log_q[i])):
                
                weight = np.exp(log_p[i] - log_r[i])
                value = log_p[i] - log_q[i]
                
                particles.append({
                    'text': texts[i],
                    'weight': weight,
                    'val': value
                })
        
        print(f"    Batch {batch_idx+1}/{n_batches} completed ({len(particles)} particles)", end='\r')
    
    print()
    
    if not particles:
        print(f"Warning: No valid particles collected for {proposal_name}")
        return {
            'proposal_name': proposal_name,
            'prompt': prompt,
            'temperature': temperature,
            'n_reps': 0,
            'kl_estimate': 0.0,
            'variance': float('inf'),
            'sample_var': float('inf'),
            'effective_sample_size': 0.0,
            'averaging_results': {},
            'n_particles': 0
        }
    
    weights = np.array([p['weight'] for p in particles])
    values = np.array([p['val'] for p in particles])
    N = len(particles)
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum == 0:
        kl_est = 0.0
        variance = float('inf')
        ess = 0.0
    else:
        norm_weights = weights / w_sum
        kl_est = np.sum(norm_weights * values)
        
        # Compute variance using the normalized importance sampling formula
        weighted_sq_errors = (norm_weights ** 2) * ((values - kl_est) ** 2)
        variance = np.sum(weighted_sq_errors) 
        
        # Effective sample size
        ess = 1.0 / np.sum(norm_weights ** 2)

    if ess > 1:
        correction_factor = ess / (ess - 1)
        sample_var = correction_factor * variance 
    else: 
        sample_var = float('inf')
    
    print(f"\nKL_estimate: {float(kl_est):.6f}")
    print(f"Asymptotic Variance: {float(variance):.6e}")
    print(f"Sample Variance: {float(sample_var):.6e}")
    print(f"Effective Sample Size: {float(ess):.1f}/{N}")
    
    averaging_results = compute_kl_with_different_averaging(particles)
    
    return {
        'proposal_name': proposal_name,
        'prompt': prompt,
        'temperature': temperature,
        'n_reps': len(particles),
        'kl_estimate': float(kl_est),
        'variance': float(variance),
        'sample_var': float(sample_var),
        'effective_sample_size': float(ess),
        'averaging_results': averaging_results,
        'n_particles': N
    }

def run_experiment(
    proposal_func,
    proposal_name: str,
    prompt: str,
    temperature: float,
    max_new_tokens: int = 20,
    n_reps: int = 200
) -> Dict:
    """Run a single experiment with given parameters"""
    
    print(f"  Testing {proposal_name} at temp={temperature} with prompt: '{prompt[:30]}...'")
    
    particles = []
    kl_sum = 0.0
    w_sum = 0.0
    
    for k in range(n_reps):
        text, lg_r, lg_p, lg_q = sample_once(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            next_token_weight=proposal_func
        )
        
        weight = math.exp(lg_p - lg_r)
        particles.append({
            'text': text,
            'weight': weight,
            'val': lg_p - lg_q
        })
        
        kl_sum += weight * (lg_p - lg_q)
        w_sum += weight
        
        if (k + 1) % 50 == 0:
            print(f"    Completed {k + 1}/{n_reps} reps", end='\r')
    
    print(f"    Completed {n_reps}/{n_reps} reps")
    
    if w_sum == 0:
        kl_est = 0.0
        variance = float('inf')
    else:
        kl_est = kl_sum / w_sum
        variance = compute_variance(particles, kl_est, w_sum)
    
    ess = compute_effective_sample_size(particles)

    print(f"\nKL_estimate: {float(kl_est)}")
    print(f"\nVariance: {float(variance)}")
    
    # Different averaging methods
    averaging_results = compute_kl_with_different_averaging(particles)
    
    return {
        'proposal_name': proposal_name,
        'prompt': prompt,
        'temperature': temperature,
        'n_reps': n_reps,
        'kl_estimate': float(kl_est),
        'variance': float(variance),
        'effective_sample_size': float(ess),
        'averaging_results': averaging_results,
        'n_particles': len(particles)
    }

def rank_proposals(all_results: Dict) -> Dict:
    """Rank proposals based on performance metrics"""
    
    # Collect all results into a flat list
    flat_results = []
    for proposal_name, experiments in all_results.items():
        for exp in experiments:
            flat_results.append({
                'proposal_name': proposal_name,
                **exp
            })
    
    # Group by proposal for overall ranking
    proposal_stats = {}
    for proposal_name in all_results.keys():
        proposal_exps = [r for r in flat_results if r['proposal_name'] == proposal_name]
        
        if not proposal_exps:
            continue
            
        # Average across all experiments for this proposal
        avg_kl = np.mean([exp['kl_estimate'] for exp in proposal_exps])
        avg_var = np.mean([exp['variance'] for exp in proposal_exps])
        avg_ess = np.mean([exp['effective_sample_size'] for exp in proposal_exps])
        avg_sample_var = np.mean([exp['sample_var'] for exp in proposal_exps])
        
        # Stability score (lower variance and higher ESS is better)
        stability_score = avg_ess / (avg_var + 1e-10)
        
        proposal_stats[proposal_name] = {
            'avg_kl': float(avg_kl),
            'avg_variance': float(avg_var),
            'avg_sample_variance': float(avg_sample_var),
            'avg_effective_sample_size': float(avg_ess),
            'stability_score': float(stability_score),
            'n_experiments': len(proposal_exps)
        }
    
    # Create rankings
    rankings = {
        'by_variance': sorted(
            proposal_stats.items(),
            key=lambda x: x[1]['avg_variance']
        ),
        'by_sample_variance': sorted(
            proposal_stats.items(),
            key=lambda x: x[1]['avg_sample_variance']
        ),
        'by_effective_sample_size': sorted(
            proposal_stats.items(),
            key=lambda x: x[1]['avg_effective_sample_size'],
            reverse=True  # Higher ESS is better
        ),
        'by_stability': sorted(
            proposal_stats.items(),
            key=lambda x: x[1]['stability_score'],
            reverse=True  # Higher stability is better
        )
    }
    
    return {
        'proposal_stats': proposal_stats,
        'rankings': rankings,
        'all_results': all_results
    }

def main():
    """Main function to run comprehensive experiments"""
    
    # Define all proposal functions to test
    proposals = {
        'just_p': just_p,
#        'eps_01': eps_01,
#        'eps_03': eps_03,
        'eps_05': eps_05,
        'eps_09': eps_09,
#        'balanced': balanced,
        'adaptive': adaptive,
        'mix09': mix09,
#        'mix03': mix03,
#        'balanced2': balanced2,
#        'geometric': geometric,
#        'exponential_family_0.5': lambda p, l_p, l_q, lg_p, lg_q: exponential_family(p, l_p, l_q, lg_p, lg_q, beta=0.5),
#        'exponential_family_1.0': lambda p, l_p, l_q, lg_p, lg_q: exponential_family(p, l_p, l_q, lg_p, lg_q, beta=1.0),
        'cross_entropy_0.5': lambda p, l_p, l_q, lg_p, lg_q: cross_entropy(p, l_p, l_q, lg_p, lg_q, beta=0.5),
        'cross_entropy_1.0': lambda p, l_p, l_q, lg_p, lg_q: cross_entropy(p, l_p, l_q, lg_p, lg_q, beta=1.0),
    }
    
    # Define experimental parameters
    temperatures = [0.4, 1.0]
    max_new_tokens = 20
    n_reps = 200  # Reduced for speed; increase for better statistics
    
    # Select subset of prompts for testing (to keep runtime reasonable)
    test_prompts = []
    for category in ['factual']:
        test_prompts.extend(PROMPT_CATEGORIES[category][:1])  # 1 prompt from each category
    
    print(f"Running comprehensive experiments with {len(proposals)} proposals")
    print(f"Temperatures: {temperatures}")
    print(f"Prompts: {test_prompts}")
    print(f"Replications per condition: {n_reps}")
    print("=" * 80)
    
    # Store all results
    all_results = {name: [] for name in proposals.keys()}
    
    # Run experiments
    total_experiments = len(proposals) * len(temperatures) * len(test_prompts)
    experiment_count = 0
    '''
    # ===== 1. COMPUTE BASELINE FIRST =====
    print("=" * 80)
    print("COMPUTING RAO-BLACKWELLIZED BASELINE ESTIMATE")
    print("=" * 80)
    
    baseline_results = {}
    for prompt in test_prompts[:2]:  # Compute baseline for first 2 prompts
        print(f"\nPrompt: '{prompt}'")
        mean_kl, variance, kls = compute_baseline_RB(
            prompt=prompt,
            n_runs=200,  
            max_new_tokens=max_new_tokens,
            temperature=0.4  # Use a standard temperature
        )
        
        baseline_results[prompt] = {
            'mean_kl': mean_kl,
            'variance': variance,
            'std_error': np.sqrt(variance / len(kls)),
            'n_runs': len(kls)
        }
        
    with open('baseline_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)

    '''
    # ===== 2. RUN IMPORTANCE SAMPLING EXPERIMENTS =====
    print("\n" + "=" * 80)
    print("RUNNING IMPORTANCE SAMPLING EXPERIMENTS")
    print("=" * 80)
    
    for proposal_name, proposal_func in proposals.items():
        print(f"\n{'='*60}")
        print(f"Testing proposal: {proposal_name}")
        print(f"{'='*60}")
        proposal_results = [] 
        
        for temperature in temperatures:
            for prompt in test_prompts:
                experiment_count += 1
                print(f"\nExperiment {experiment_count}/{total_experiments}")
                
                # Run the experiment
                result = run_batch_experiment(
                    proposal_func=proposal_func,
                    proposal_name=proposal_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    n_reps=n_reps,
                    batch_size=32
                )
                
                # Store result
                all_results[proposal_name].append(result)
                proposal_results.append(result)
                
        with open(f'results_{proposal_name}.json', 'w') as f:
            json.dump(proposal_results, f, indent=2, default=str)

        print(f"✓ Saved {len(proposal_results)} experiments to results_{proposal_name}.json")
            
    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"Total experiments run: {experiment_count}")
    print(f"Saving results...")
    
    # Rank proposals
    rankings = rank_proposals(all_results)
    
    # Save results
    output_data = {
        'experiment_config': {
            'temperatures': temperatures,
            'max_new_tokens': max_new_tokens,
            'n_reps': n_reps,
            'test_prompts': test_prompts,
            'proposals_tested': list(proposals.keys())
        },
        'all_results': all_results,
        'rankings': rankings
    }
    
    # Save to file
    with open('comprehensive_kl_results.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print("Results saved to 'comprehensive_kl_results.json'")

    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE")
    print("=" * 80)

    with open('baseline_results.json', 'r') as f:
        baseline_results = json.load(f)

    for prompt, baseline in baseline_results.items():
        print(f"\nBaseline for '{prompt[:30]}...':")
        print(f"  KL estimate: {baseline['mean_kl']:.6f}")
        print(f"  Variance: {baseline['variance']:.6f}")
        print(f"  Std Error: {baseline['std_error']:.6f}")
    
    # Print summary
    print("\n" + "="*80)
    print("RANKINGS SUMMARY")
    print("="*80)
    
    print("\nTop 5 by Variance (lower is better):")
    for i, (name, stats) in enumerate(rankings['rankings']['by_sample_variance'][:5], 1):
        print(f"{i}. {name}: variance={stats['avg_sample_variance']:.6f}, ESS={stats['avg_effective_sample_size']:.1f}")
    
    print("\nTop 5 by Effective Sample Size (higher is better):")
    for i, (name, stats) in enumerate(rankings['rankings']['by_effective_sample_size'][:5], 1):
        print(f"{i}. {name}: ESS={stats['avg_effective_sample_size']:.1f}, variance={stats['avg_variance']:.6f}")
    
    print("\nTop 5 by Stability Score (higher is better):")
    for i, (name, stats) in enumerate(rankings['rankings']['by_stability'][:5], 1):
        print(f"{i}. {name}: stability={stats['stability_score']:.6f}, ESS={stats['avg_effective_sample_size']:.1f}, var={stats['avg_variance']:.6f}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # You can still run individual tests via command line if needed
    if len(sys.argv) > 1:
        # Run individual proposal test (backward compatibility)
        n_reps = 1000
        t_length = 20
        for ff in sys.argv[1:]:
            # Simplified run for backward compatibility
            particles = []
            kl_sum = 0
            w_sum = 0
            for k in range(n_reps):
                text, lg_r, lg_p, lg_q = sample_once(
                    "An interesting fact is that",
                    max_new_tokens=t_length,
                    temperature=0.4,
                    next_token_weight=eval(ff)
                )
                weight = math.exp(lg_p - lg_r)
                particles.append({
                    'text': text,
                    'weight': weight,
                    'val': lg_p - lg_q
                })
                kl_sum += weight * (lg_p - lg_q)
                w_sum += weight
                print(f"KL: {kl_sum / w_sum if w_sum > 0 else 0}\t\t", end='\r')
            
            kl = kl_sum / w_sum if w_sum > 0 else 0
            print(f"\n{ff}: KL estimate: {kl:0.6f}")
    else:
        # Run comprehensive experiments
        main()
