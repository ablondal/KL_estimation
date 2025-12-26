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

def compute_variance(particles: List[Dict], kl_est: float, w_sum: float) -> float:
    """Compute variance of KL estimate"""
    var_sum = 0.0
    for p in particles:
        var_sum += p['weight'] * (p['val'] - kl_est) * (p['val'] - kl_est)
    return var_sum / w_sum if w_sum > 0 else float('inf')

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
    
    return {
        'standard': float(kl_standard),
        'clipped': float(kl_clipped),
        'bayesian': float(kl_bayesian),
        'bootstrap_ci': (float(ci_lower), float(ci_upper)),
        'bootstrap_mean': float(np.mean(bootstrap_estimates)) if bootstrap_estimates else 0.0,
        'weight_entropy': float(weight_entropy)
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
    
    # Compute effective sample size
    weights = [p['weight'] for p in particles]
    if sum(weights) > 0:
        norm_weights = [w / sum(weights) for w in weights]
        ess = 1.0 / sum(w * w for w in norm_weights)
    else:
        ess = 0.0
    
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
        
        # Stability score (lower variance and higher ESS is better)
        stability_score = avg_ess / (avg_var + 1e-10)
        
        proposal_stats[proposal_name] = {
            'avg_kl': float(avg_kl),
            'avg_variance': float(avg_var),
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
        'eps_01': eps_01,
        'eps_03': eps_03,
        'eps_05': eps_05,
        'eps_09': eps_09,
        'balanced': balanced,
        'adaptive': adaptive,
        'mix09': mix09,
        'mix03': mix03,
        'balanced2': balanced2,
        'geometric': geometric,
        'exponential_family_0.5': lambda p, l_p, l_q, lg_p, lg_q: exponential_family(p, l_p, l_q, lg_p, lg_q, beta=0.5),
        'exponential_family_1.0': lambda p, l_p, l_q, lg_p, lg_q: exponential_family(p, l_p, l_q, lg_p, lg_q, beta=1.0),
        'cross_entropy_0.5': lambda p, l_p, l_q, lg_p, lg_q: cross_entropy(p, l_p, l_q, lg_p, lg_q, beta=0.5),
        'cross_entropy_1.0': lambda p, l_p, l_q, lg_p, lg_q: cross_entropy(p, l_p, l_q, lg_p, lg_q, beta=1.0),
    }
    
    # Define experimental parameters
    temperatures = [0.1, 0.4, 1.0, 2.0]
    max_new_tokens = 20
    n_reps = 200  # Reduced for speed; increase for better statistics
    
    # Select subset of prompts for testing (to keep runtime reasonable)
    test_prompts = []
    for category in ['factual', 'creative']:
        test_prompts.extend(PROMPT_CATEGORIES[category][:2])  # 2 prompts from each category
    
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
    
    for proposal_name, proposal_func in proposals.items():
        print(f"\n{'='*60}")
        print(f"Testing proposal: {proposal_name}")
        print(f"{'='*60}")
        
        for temperature in temperatures:
            for prompt in test_prompts:
                experiment_count += 1
                print(f"\nExperiment {experiment_count}/{total_experiments}")
                
                # Run the experiment
                result = run_experiment(
                    proposal_func=proposal_func,
                    proposal_name=proposal_name,
                    prompt=prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    n_reps=n_reps
                )
                
                # Store result
                all_results[proposal_name].append(result)
    
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
    
    # Print summary
    print("\n" + "="*80)
    print("RANKINGS SUMMARY")
    print("="*80)
    
    print("\nTop 5 by Variance (lower is better):")
    for i, (name, stats) in enumerate(rankings['rankings']['by_variance'][:5], 1):
        print(f"{i}. {name}: variance={stats['avg_variance']:.6f}, ESS={stats['avg_effective_sample_size']:.1f}")
    
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
