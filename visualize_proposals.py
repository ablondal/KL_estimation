"""
Visualization and Diagnostics for Proposal Distributions
Analyzes different proposal distributions for KL divergence estimation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def plot_distribution_overlay(P_probs, Q_probs, r_probs_dict, max_tokens=200):
    """
    Overlay plot: P(x), Q(x), r(x), and |log(P/Q)| on same axes
    
    Args:
        P_probs: Base model probabilities 
        Q_probs: Fine-tuned model probabilities 
        r_probs_dict: Dict of {'name': name, 'r': r_probs} for different proposals
        max_tokens: Number of tokens to show 
    """
    n_proposals = len(r_probs_dict)
    fig, axes = plt.subplots(n_proposals + 1, 1, figsize=(15, 4 * (n_proposals + 1)))
    
    if n_proposals == 0:
        axes = [axes]
    
    # Truncate for visualization
    tokens = np.arange(min(max_tokens, len(P_probs)))
    P_viz = P_probs[:max_tokens]
    Q_viz = Q_probs[:max_tokens]
    
    # Compute |log(P/Q)|
    log_ratio = np.abs(np.log(np.clip(P_viz, 1e-12, 1.0) / np.clip(Q_viz, 1e-12, 1.0)))
    log_ratio_scaled = log_ratio / log_ratio.max()  # Scale for visualization
    
    # Plot P, Q, and |log(P/Q)| on first subplot
    ax = axes[0]
    ax.plot(tokens, P_viz, label='P(x) - Base', linewidth=2, alpha=0.8)
    ax.plot(tokens, Q_viz, label='Q(x) - Fine-tuned', linewidth=2, alpha=0.8)
    ax.plot(tokens, log_ratio_scaled * P_viz.max(), label='|log(P/Q)| (scaled)', 
            linewidth=2, alpha=0.6, linestyle='--', color='red')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Probability')
    ax.set_title('Base Distributions: P(x), Q(x), and |log(P/Q)|', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot each proposal distribution
    for idx, proposal in enumerate(r_probs_dict):
        ax = axes[idx + 1]
        r_viz = proposal['r'][:max_tokens]
        
        ax.plot(tokens, P_viz, label='P(x)', linewidth=2, alpha=0.6)
        ax.plot(tokens, Q_viz, label='Q(x)', linewidth=2, alpha=0.6)
        ax.plot(tokens, r_viz, label=f"r(x) - {proposal['name']}", 
                linewidth=3, alpha=0.9, color='green')
        ax.plot(tokens, log_ratio_scaled * P_viz.max(), label='|log(P/Q)| (scaled)', 
                linewidth=1, alpha=0.4, linestyle='--', color='red')
        
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Probability')
        ax.set_title(f"Importance Sampling Proposal: {proposal['name']}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_importance_weights(P_probs, Q_probs, r_probs_dict, n_samples=10000):
    """
    Histogram of importance weights P(x)/r(x) to spot exploding weights
    
    Args:
        P_probs: Base model probabilities
        Q_probs: Fine-tuned model probabilities
        r_probs_dict: List of {'name': name, 'r': r_probs} for different proposals
        n_samples: Number of samples to draw for histogram
    """
    n_proposals = len(r_probs_dict)
    fig, axes = plt.subplots(2, (n_proposals + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, proposal in enumerate(r_probs_dict):
        ax = axes[idx]
        r_probs = proposal['r']
        
        # Sample from r
        samples = np.random.choice(len(r_probs), size=n_samples, p=r_probs)
        
        # Compute importance weights
        weights = P_probs[samples] / r_probs[samples]
        
        # Compute ESS
        ess = (weights.sum() ** 2) / (weights ** 2).sum()
        ess_ratio = ess / n_samples
        
        # Plot histogram
        ax.hist(weights, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(weights.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {weights.mean():.2f}')
        ax.axvline(np.median(weights), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(weights):.2f}')
        
        ax.set_xlabel('Importance Weight (P/r)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f"{proposal['name']}\nESS: {ess:.0f} ({ess_ratio:.1%})", 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add text with statistics
        stats_text = f"Max: {weights.max():.2f}\nStd: {weights.std():.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(len(r_probs_dict), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_coverage_map(P_probs, Q_probs, r_probs_dict, n_samples=10000, n_regions=10):
    """
    Shows which vocab regions get sampled under different r's
    
    Args:
        P_probs: Base model probabilities
        Q_probs: Fine-tuned model probabilities
        r_probs_dict: List of {'name': name, 'r': r_probs}
        n_samples: Number of samples to draw
        n_regions: Number of vocab regions to split into
    """
    vocab_size = len(P_probs)
    region_size = vocab_size // n_regions
    
    n_proposals = len(r_probs_dict)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data
    coverage_data = np.zeros((n_proposals + 1, n_regions))  # +1 for P itself
    region_labels = [f"{i*region_size}-{(i+1)*region_size-1}" for i in range(n_regions)]
    
    # Sample from P (baseline)
    samples_P = np.random.choice(vocab_size, size=n_samples, p=P_probs)
    for i in range(n_regions):
        start = i * region_size
        end = (i + 1) * region_size
        coverage_data[0, i] = np.sum((samples_P >= start) & (samples_P < end)) / n_samples
    
    # Sample from each proposal
    for idx, proposal in enumerate(r_probs_dict):
        r_probs = proposal['r']
        samples = np.random.choice(vocab_size, size=n_samples, p=r_probs)
        
        for i in range(n_regions):
            start = i * region_size
            end = (i + 1) * region_size
            coverage_data[idx + 1, i] = np.sum((samples >= start) & (samples < end)) / n_samples
    
    # Plot 1: Heatmap of coverage
    ax1 = axes[0]
    method_names = ['P (Naive MC)'] + [p['name'] for p in r_probs_dict]
    im = ax1.imshow(coverage_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_xticks(range(n_regions))
    ax1.set_xticklabels(region_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_yticks(range(len(method_names)))
    ax1.set_yticklabels(method_names, fontsize=9)
    ax1.set_xlabel('Vocabulary Region (Token Range)', fontsize=11)
    ax1.set_ylabel('Sampling Method', fontsize=11)
    ax1.set_title('Coverage Map: Proportion of Samples per Region', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Sampling Proportion', fontsize=10)
    
    # Add text annotations
    for i in range(len(method_names)):
        for j in range(n_regions):
            text = ax1.text(j, i, f'{coverage_data[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    # Plot 2: Comparison of tail coverage
    ax2 = axes[1]
    tail_start = vocab_size // 2  # Define "tail" as second half
    
    tail_coverage = []
    for idx in range(len(method_names)):
        if idx == 0:
            samples = samples_P
        else:
            samples = np.random.choice(vocab_size, size=n_samples, p=r_probs_dict[idx-1]['r'])
        tail_coverage.append(np.sum(samples >= tail_start) / n_samples)
    
    bars = ax2.bar(range(len(method_names)), tail_coverage, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(method_names)))
    ax2.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Proportion in Tail (2nd Half)', fontsize=11)
    ax2.set_title('Tail Coverage Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, tail_coverage):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def compute_ess_metrics(P_probs, r_probs_dict, n_samples=10000):
    """
    Compute Effective Sample Size (ESS) for different proposals
    
    ESS = (Σw)² / Σw²
    
    Returns dict with ESS statistics
    """
    ess_results = []
    
    for proposal in r_probs_dict:
        r_probs = proposal['r']
        
        # Sample from r
        samples = np.random.choice(len(r_probs), size=n_samples, p=r_probs)
        
        # Compute importance weights
        weights = P_probs[samples] / r_probs[samples]
        
        # Compute ESS
        ess = (weights.sum() ** 2) / (weights ** 2).sum()
        ess_ratio = ess / n_samples
        
        ess_results.append({
            'name': proposal['name'],
            'ess': ess,
            'ess_ratio': ess_ratio,
            'max_weight': weights.max(),
            'mean_weight': weights.mean(),
            'std_weight': weights.std()
        })
    
    return ess_results


def plot_ess_comparison(ess_results):
    """
    Plot ESS comparison across different proposals
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    names = [r['name'] for r in ess_results]
    ess_ratios = [r['ess_ratio'] for r in ess_results]
    max_weights = [r['max_weight'] for r in ess_results]
    
    # Plot 1: ESS Ratio
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(names)), ess_ratios, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('ESS Ratio (ESS / n_samples)', fontsize=11)
    ax1.set_title('Effective Sample Size Comparison', fontsize=12, fontweight='bold')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect (ESS=n)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, ess_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2%}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Max Weight (indicator of stability)
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(names)), max_weights, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Maximum Importance Weight', fontsize=11)
    ax2.set_title('Weight Stability (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, max_weights):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_comprehensive_report(P_probs, Q_probs, r_probs_dict, n_samples=10000):
    """
    Generate all visualization plots and save them
    """
    print("=" * 80)
    print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
    print("=" * 80)
    
    # Create plots directory in current working directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    plots_dir = os.path.join(os.getcwd(), 'plots', timestamp)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\nPlots will be saved to: {plots_dir}")
    
    # 1. Distribution Overlay
    print("\n[1/5] Creating distribution overlay plots...")
    fig1 = plot_distribution_overlay(P_probs, Q_probs, r_probs_dict, max_tokens=200)
    plt.savefig(os.path.join(plots_dir, '01_distribution_overlay.png'), dpi=150, bbox_inches='tight')
    print("    ✓ Saved: 01_distribution_overlay.png")
    
    # 2. Importance Weights
    print("[2/5] Creating importance weight histograms...")
    fig2 = plot_importance_weights(P_probs, Q_probs, r_probs_dict, n_samples)
    plt.savefig(os.path.join(plots_dir, '02_importance_weights.png'), dpi=150, bbox_inches='tight')
    print("    ✓ Saved: 02_importance_weights.png")
    
    # 3. Coverage Map
    print("[3/5] Creating coverage map...")
    fig3 = plot_coverage_map(P_probs, Q_probs, r_probs_dict, n_samples, n_regions=10)
    plt.savefig(os.path.join(plots_dir, '03_coverage_map.png'), dpi=150, bbox_inches='tight')
    print("    ✓ Saved: 03_coverage_map.png")
    
    # 4. ESS Analysis
    print("[4/5] Computing ESS metrics...")
    ess_results = compute_ess_metrics(P_probs, r_probs_dict, n_samples)
    
    print("\nESS Summary:")
    print("-" * 80)
    print(f"{'Method':<25} {'ESS Ratio':>12} {'Max Weight':>12} {'Mean Weight':>12}")
    print("-" * 80)
    for result in ess_results:
        print(f"{result['name']:<25} {result['ess_ratio']:>12.2%} "
              f"{result['max_weight']:>12.2f} {result['mean_weight']:>12.2f}")
    
    # 5. ESS Comparison Plot
    print("\n[5/5] Creating ESS comparison plot...")
    fig4 = plot_ess_comparison(ess_results)
    plt.savefig(os.path.join(plots_dir, '04_ess_comparison.png'), dpi=150, bbox_inches='tight')
    print("    ✓ Saved: 04_ess_comparison.png")
    
    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print(f"All plots saved to: {plots_dir}")
    print("=" * 80)
    
    plt.close('all')  # Close all figures to free memory
    
    return ess_results
