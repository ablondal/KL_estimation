import numpy as np
import torch
import math
import json
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any, Tuple
from results.all_results import all_results

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

rankings = rank_proposals(all_results)

# Print temperature labels for each experiment
print("\n" + "="*80)
print("TEMPERATURE ANALYSIS")
print("="*80)

for proposal_name, experiments in all_results.items():
    print(f"\n{proposal_name}:")
    for exp in experiments:
        temp_label = f"temp={exp['temperature']:.1f}"
        print(f"  {temp_label}: KL={exp['kl_estimate']:.4f}, ESS={exp['effective_sample_size']:.1f}, "
              f"var={exp['variance']:.4f}, sample_var={exp['sample_var']:.4f}")

print("\n" + "="*80)
print("TOP RANKINGS")
print("="*80)

print("\nTop 5 by Sample Variance (lower is better):")
for i, (name, stats) in enumerate(rankings['rankings']['by_sample_variance'][:5], 1):
    print(f"{i}. {name}: sample_variance={stats['avg_sample_variance']:.6f}, "
          f"ESS={stats['avg_effective_sample_size']:.1f}")

print("\nTop 5 by Effective Sample Size (higher is better):")
for i, (name, stats) in enumerate(rankings['rankings']['by_effective_sample_size'][:5], 1):
    print(f"{i}. {name}: ESS={stats['avg_effective_sample_size']:.1f}, "
          f"variance={stats['avg_variance']:.6f}")

print("\nTop 5 by Stability Score (higher is better):")
for i, (name, stats) in enumerate(rankings['rankings']['by_stability'][:5], 1):
    print(f"{i}. {name}: stability={stats['stability_score']:.6f}, "
          f"ESS={stats['avg_effective_sample_size']:.1f}, var={stats['avg_variance']:.6f}")

# Temperature comparison across proposals
print("\n" + "="*80)
print("TEMPERATURE PERFORMANCE COMPARISON")
print("="*80)

# Group by temperature
temp_results = {}
for proposal_name, experiments in all_results.items():
    for exp in experiments:
        temp = exp['temperature']
        if temp not in temp_results:
            temp_results[temp] = []
        temp_results[temp].append({
            'proposal': proposal_name,
            'kl': exp['kl_estimate'],
            'ess': exp['effective_sample_size'],
            'var': exp['variance']
        })

for temp in sorted(temp_results.keys()):
    print(f"\nTemperature {temp:.1f}:")
    temp_exps = temp_results[temp]
    
    # Sort by KL (lower is better)
    sorted_by_kl = sorted(temp_exps, key=lambda x: x['kl'])
    print(f"  Best KL: {sorted_by_kl[0]['proposal']} ({sorted_by_kl[0]['kl']:.4f})")
    print(f"  Worst KL: {sorted_by_kl[-1]['proposal']} ({sorted_by_kl[-1]['kl']:.4f})")
    
    # Sort by ESS (higher is better)
    sorted_by_ess = sorted(temp_exps, key=lambda x: x['ess'], reverse=True)
    print(f"  Best ESS: {sorted_by_ess[0]['proposal']} ({sorted_by_ess[0]['ess']:.1f})")
    print(f"  Worst ESS: {sorted_by_ess[-1]['proposal']} ({sorted_by_ess[-1]['ess']:.1f})")
    
    # Sort by variance (lower is better)
    sorted_by_var = sorted(temp_exps, key=lambda x: x['var'])
    print(f"  Best variance: {sorted_by_var[0]['proposal']} ({sorted_by_var[0]['var']:.4f})")
    print(f"  Worst variance: {sorted_by_var[-1]['proposal']} ({sorted_by_var[-1]['var']:.4f})")
