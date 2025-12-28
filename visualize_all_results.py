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

print("\nTop 5 by Variance (lower is better):")
for i, (name, stats) in enumerate(rankings['rankings']['by_sample_variance'][:5], 1):
    print(f"{i}. {name}: variance={stats['avg_sample_variance']:.6f}, ESS={stats['avg_effective_sample_size']:.1f}")

print("\nTop 5 by Effective Sample Size (higher is better):")
for i, (name, stats) in enumerate(rankings['rankings']['by_effective_sample_size'][:5], 1):
    print(f"{i}. {name}: ESS={stats['avg_effective_sample_size']:.1f}, variance={stats['avg_variance']:.6f}")

print("\nTop 5 by Stability Score (higher is better):")
for i, (name, stats) in enumerate(rankings['rankings']['by_stability'][:5], 1):
    print(f"{i}. {name}: stability={stats['stability_score']:.6f}, ESS={stats['avg_effective_sample_size']:.1f}, var={stats['avg_variance']:.6f}")

rankings = rank_proposals(all_results)


all_results = {'just_p': [{'proposal_name': 'just_p', 'prompt': 'An interesting fact is that', 'temperature': 0.4, 'n_reps': 200, 'kl_estimate': 11.181014060974121, 'variance': 0.31894630193710327, 'sample_var': 0.32054904103279114, 'effective_sample_size': 200.0, 'averaging_results': {'standard': 11.181015014648438, 'clipped': 11.181014060974121, 'bayesian': 11.03918170928955, 'bootstrap_ci': (10.210012435913086, 12.287035942077637), 'bootstrap_mean': 11.180739402770996, 'weight_entropy': 5.298317909240723}, 'n_particles': 200}, {'proposal_name': 'just_p', 'prompt': 'An interesting fact is that', 'temperature': 1.0, 'n_reps': 200, 'kl_estimate': 7.517736434936523, 'variance': 0.12805509567260742, 'sample_var': 0.12869858741760254, 'effective_sample_size': 199.99996948242188, 'averaging_results': {'standard': 7.517735958099365, 'clipped': 7.517739772796631, 'bayesian': 7.444820404052734, 'bootstrap_ci': (6.7806854248046875, 8.298650741577148), 'bootstrap_mean': 7.538107872009277, 'weight_entropy': 5.298317909240723}, 'n_particles': 200}], 'eps_05': [{'proposal_name': 'eps_05', 'prompt': 'An interesting fact is that', 'temperature': 0.4, 'n_reps': 200, 'kl_estimate': 11.319223403930664, 'variance': 0.49997371435165405, 'sample_var': 0.5058059096336365, 'effective_sample_size': 86.72631072998047, 'averaging_results': {'standard': 11.319225311279297, 'clipped': 11.507786750793457, 'bayesian': 11.228927612304688, 'bootstrap_ci': (9.81149959564209, 12.71921443939209), 'bootstrap_mean': 11.508387565612793, 'weight_entropy': 4.752965927124023}, 'n_particles': 200}, {'proposal_name': 'eps_05', 'prompt': 'An interesting fact is that', 'temperature': 1.0, 'n_reps': 200, 'kl_estimate': 8.407246589660645, 'variance': 0.6110739707946777, 'sample_var': 0.6212035417556763, 'effective_sample_size': 61.32593536376953, 'averaging_results': {'standard': 8.407247543334961, 'clipped': 8.558365821838379, 'bayesian': 8.474954605102539, 'bootstrap_ci': (6.976376056671143, 9.854504585266113), 'bootstrap_mean': 8.474865913391113, 'weight_entropy': 4.75405216217041}, 'n_particles': 200}], 'eps_09': [{'proposal_name': 'eps_09', 'prompt': 'An interesting fact is that', 'temperature': 0.4, 'n_reps': 200, 'kl_estimate': 10.802488327026367, 'variance': 0.26579761505126953, 'sample_var': 0.2672653794288635, 'effective_sample_size': 182.09072875976562, 'averaging_results': {'standard': 10.802488327026367, 'clipped': 10.805891036987305, 'bayesian': 10.747413635253906, 'bootstrap_ci': (9.716842651367188, 11.617934226989746), 'bootstrap_mean': 10.767059326171875, 'weight_entropy': 5.239609718322754}, 'n_particles': 200}, {'proposal_name': 'eps_09', 'prompt': 'An interesting fact is that', 'temperature': 1.0, 'n_reps': 200, 'kl_estimate': 7.945351600646973, 'variance': 0.13870590925216675, 'sample_var': 0.13944774866104126, 'effective_sample_size': 187.97238159179688, 'averaging_results': {'standard': 7.9453511238098145, 'clipped': 8.014531135559082, 'bayesian': 7.931687355041504, 'bootstrap_ci': (7.291621685028076, 8.808390617370605), 'bootstrap_mean': 7.977961540222168, 'weight_entropy': 5.268685340881348}, 'n_particles': 200}], 'adaptive': [{'proposal_name': 'adaptive', 'prompt': 'An interesting fact is that', 'temperature': 0.4, 'n_reps': 200, 'kl_estimate': 11.387930870056152, 'variance': 0.6871041059494019, 'sample_var': 0.6936402320861816, 'effective_sample_size': 106.12422180175781, 'averaging_results': {'standard': 11.387930870056152, 'clipped': 11.784945487976074, 'bayesian': 11.526996612548828, 'bootstrap_ci': (9.789908409118652, 12.607430458068848), 'bootstrap_mean': 11.404023170471191, 'weight_entropy': 4.975892066955566}, 'n_particles': 200}, {'proposal_name': 'adaptive', 'prompt': 'An interesting fact is that', 'temperature': 1.0, 'n_reps': 200, 'kl_estimate': 8.1085786819458, 'variance': 0.3084946572780609, 'sample_var': 0.3122171461582184, 'effective_sample_size': 83.87339782714844, 'averaging_results': {'standard': 8.1085786819458, 'clipped': 8.611456871032715, 'bayesian': 8.25128173828125, 'bootstrap_ci': (6.818564414978027, 9.013513565063477), 'bootstrap_mean': 8.03683090209961, 'weight_entropy': 4.926621913909912}, 'n_particles': 200}], 'mix09': [{'proposal_name': 'mix09', 'prompt': 'An interesting fact is that', 'temperature': 0.4, 'n_reps': 200, 'kl_estimate': 11.383373260498047, 'variance': 0.5413641333580017, 'sample_var': 0.5449060201644897, 'effective_sample_size': 153.8449249267578, 'averaging_results': {'standard': 11.38337230682373, 'clipped': 11.102449417114258, 'bayesian': 11.030698776245117, 'bootstrap_ci': (10.18689250946045, 12.72331714630127), 'bootstrap_mean': 11.426417350769043, 'weight_entropy': 5.113326549530029}, 'n_particles': 200}, {'proposal_name': 'mix09', 'prompt': 'An interesting fact is that', 'temperature': 1.0, 'n_reps': 200, 'kl_estimate': 8.246613502502441, 'variance': 0.38420432806015015, 'sample_var': 0.38655561208724976, 'effective_sample_size': 164.4042510986328, 'averaging_results': {'standard': 8.246613502502441, 'clipped': 8.004375457763672, 'bayesian': 8.021425247192383, 'bootstrap_ci': (7.091350555419922, 9.405089378356934), 'bootstrap_mean': 8.206940650939941, 'weight_entropy': 5.179025650024414}, 'n_particles': 200}], 'cross_entropy_0.5': [{'proposal_name': 'cross_entropy_0.5', 'prompt': 'An interesting fact is that', 'temperature': 0.4, 'n_reps': 200, 'kl_estimate': 9.495841979980469, 'variance': 0.5907349586486816, 'sample_var': 0.6928909420967102, 'effective_sample_size': 6.782675743103027, 'averaging_results': {'standard': 9.495841026306152, 'clipped': 10.817285537719727, 'bayesian': 9.73159122467041, 'bootstrap_ci': (8.63305377960205, 11.352396011352539), 'bootstrap_mean': 9.839064598083496, 'weight_entropy': 3.8341753482818604}, 'n_particles': 200}, {'proposal_name': 'cross_entropy_0.5', 'prompt': 'An interesting fact is that', 'temperature': 1.0, 'n_reps': 200, 'kl_estimate': 7.566459655761719, 'variance': 0.16203275322914124, 'sample_var': 0.16306640207767487, 'effective_sample_size': 157.75888061523438, 'averaging_results': {'standard': 7.566459655761719, 'clipped': 7.465120792388916, 'bayesian': 7.566594123840332, 'bootstrap_ci': (6.785248279571533, 8.252582550048828), 'bootstrap_mean': 7.510338306427002, 'weight_entropy': 5.192865371704102}, 'n_particles': 200}], 'cross_entropy_1.0': [{'proposal_name': 'cross_entropy_1.0', 'prompt': 'An interesting fact is that', 'temperature': 0.4, 'n_reps': 200, 'kl_estimate': 11.102648735046387, 'variance': 0.9884322285652161, 'sample_var': 1.0064595937728882, 'effective_sample_size': 55.82963180541992, 'averaging_results': {'standard': 11.10264778137207, 'clipped': 12.10659122467041, 'bayesian': 11.452322959899902, 'bootstrap_ci': (9.49443531036377, 13.204549789428711), 'bootstrap_mean': 11.040631294250488, 'weight_entropy': 4.740803241729736}, 'n_particles': 200}, {'proposal_name': 'cross_entropy_1.0', 'prompt': 'An interesting fact is that', 'temperature': 1.0, 'n_reps': 200, 'kl_estimate': 8.133522033691406, 'variance': 0.35546091198921204, 'sample_var': 0.36508360505104065, 'effective_sample_size': 37.93991470336914, 'averaging_results': {'standard': 8.13352108001709, 'clipped': 7.722100257873535, 'bayesian': 8.106213569641113, 'bootstrap_ci': (7.1184821128845215, 8.855536460876465), 'bootstrap_mean': 8.013799667358398, 'weight_entropy': 4.687697410583496}, 'n_particles': 200}]
