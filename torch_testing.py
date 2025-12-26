import torch, math, json, sys
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    dtype=dtype
).to(device).eval()

model_b = AutoModelForCausalLM.from_pretrained(
    model_b_id,
    dtype=dtype
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
    log_p = 0     # sum_t log P(x_t | x_{x<t}); culmulative divergence
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
    # The epsilon is required so we don't get NaN / infs
    q = torch.exp(l_q)
    prop = torch.sqrt((p * q * torch.abs(l_p - l_q + lg_p - lg_q)) / (p + q + 1e-5))
    return torch.nn.functional.normalize(prop, p=1, dim=0)

def exponential_family(p, l_p, l_q, lg_p, lg_q, beta=1.0):
    # r ∝ P * exp(beta * |log(P/Q)|)
    log_ratio = l_p - l_q + (lg_p - lg_q)
    prop = p * torch.exp(beta * torch.abs(log_ratio))
    return torch.nn.functional.normalize(prop, p=1, dim=0)
    
def cross_entropy((p, l_p, l_q, lg_p, lg_q, beta=1.0):
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


# kls = []
# kl_sum = 0
# kl_num = 0
# for k in range(10):
#     text, kl_expectations = calc_kl_RB(
#         "An interesting fact is that",
#         max_new_tokens=30,
#         temperature=0.4
#     )
#     kl_sum += sum(kl_expectations)
#     kl_num += 1
#     kls.append(kl_expectations)
#     print(f"KL #{k}: {kl_sum / kl_num}\t\t", end='\r')
# mu = kl_sum / kl_num
# print(f"KL estimate: {mu}\t\t")
# var_sum = 0
# for kl in kls:
#     var_sum += (mu-sum(kl) )*(mu-sum(kl) )
# print(f"Var: {var_sum / kl_num}\t\t")

# with open(sys.argv[2], 'w') as f:
#     json.dump(kls, f, indent=2)

def compute_variance(particles, kl_est):
    var_sum = 0
    for p in particles:
        var_sum += p['weight'] * (p['val'] - kl_est) * (p['val'] - kl_est)
    var = var_sum / w_sum
    return var
    
def analyze_temperature_effect(func, temperatures=[0.1, 0.4, 0.7, 1.0, 1.5]):
    """Analyze how temperature affects KL estimates and variance"""
    results = {}
    for temp in temperatures:
        print(f"\n=== Temperature: {temp} ===")
        
        # Run importance sampling at this temperature
        particles = []
        kl_sum = 0
        w_sum = 0
        
        for k in range(n_reps):
            text, lg_r, lg_p, lg_q = sample_once(
                "An interesting fact is that",
                max_new_tokens=20,
                temperature=temp,
                next_token_weight=func
            )
            weight = math.exp(lg_p - lg_r)
            particles.append({
                'text': text, 
                'weight': weight,
                'val': lg_p - lg_q, 
                'temp': temp})
            kl_sum += weight * (lg_p - lg_q)
            w_sum += weight
        
        kl_est = kl_sum / w_sum
        var = compute_variance(particles, kl_est)
        
        results[temp] = {
            'kl_estimate': kl_est,
            'variance': var,
            'effective_sample_size': w_sum**2 / sum(p['weight']**2 for p in particles)
        }
    
    return results


def analyze_prompt_variance(func, n_prompts_per_category=3):
    """Test how KL estimates vary across different prompt types"""
    results_by_category = {}
    
    for category, prompts in PROMPT_CATEGORIES.items():
        category_results = []
        
        for prompt in prompts[:n_prompts_per_category]:
            # Run estimation for this prompt
            prompt_kl, prompt_var = run_smc(func, 20, 200, func_name, prompt)
            
            category_results.append({
                'prompt': prompt,
                'kl': prompt_kl,
                'variance': prompt_var
            })
        
        # Compute statistics across prompts
        avg_kl = np.mean([r['kl'] for r in category_results])
        kl_std = np.std([r['kl'] for r in category_results])
        avg_var = np.mean([r['variance'] for r in category_results])
        
        results_by_category[category] = {
            'avg_kl': avg_kl,
            'kl_std_across_prompts': kl_std,
            'avg_variance': avg_var,
            'prompt_specific_results': category_results
        }
    
    return results_by_category

def compute_kl_with_different_averaging(particles):
    """Compare different weighting/averaging strategies"""
    
    # 1. Standard importance sampling (current)
    weights = [p['weight'] for p in particles]
    values = [p['val'] for p in particles]
    
    kl_standard = np.average(values, weights=weights)
    
    # 2. Self-normalized with clipping (more robust)
    clipped_weights = np.clip(weights, np.percentile(weights, 5), 
                               np.percentile(weights, 95))
    kl_clipped = np.average(values, weights=clipped_weights)
    
    # 3. Bayesian averaging (assuming prior)
    # Assuming KL ~ Normal(μ, σ²) with weak prior
    prior_strength = 0.1
    prior_mean = np.median(values)
    
    # Weighted mean and variance
    weighted_mean = kl_standard
    weighted_var = np.average((values - weighted_mean)**2, weights=weights)
    
    # Bayesian shrinkage
    kl_bayesian = (prior_strength * prior_mean + weighted_mean) / (1 + prior_strength)
    
    # 4. Bootstrap confidence intervals
    bootstrap_estimates = []
    for _ in range(1000):
        idx = np.random.choice(len(particles), size=len(particles), replace=True)
        bs_weights = [weights[i] for i in idx]
        bs_values = [values[i] for i in idx]
        bootstrap_estimates.append(np.average(bs_values, weights=bs_weights))
    
    ci_lower = np.percentile(bootstrap_estimates, 2.5)
    ci_upper = np.percentile(bootstrap_estimates, 97.5)
    
    return {
        'standard': kl_standard,
        'clipped': kl_clipped,
        'bayesian': kl_bayesian,
        'bootstrap_ci': (ci_lower, ci_upper),
        'bootstrap_mean': np.mean(bootstrap_estimates),
        'weight_entropy': -np.sum(weights/np.sum(weights) * 
                                  np.log(weights/np.sum(weights) + 1e-10))
    }
    
def run_smc(func, length, reps, outp, prompt="An interesting fact is that"):
    print(f"Proposal distribution: {outp}\nReps, Length: {reps}, {length}")

    particles = []
    kl_sum = 0
    w_sum = 0
    for k in range(reps):
        text, lg_r, lg_p, lg_q = sample_once(
            prompt,
            max_new_tokens=length,
            temperature=0.4,
            next_token_weight = func
        )
        weight = math.exp(lg_p - lg_r)

        particles.append(
            {'text': text,
            'weight': weight,
            'val': lg_p - lg_q})
        
        kl_sum += weight * (lg_p - lg_q)
        w_sum += weight
        print(f"KL: {kl_sum / w_sum}\t\t", end='\r')

    with open('tests2/'+outp+'.json', 'w') as f:
        json.dump(particles, f, indent=2)

    kl = kl_sum / w_sum
    print(f"KL estimate: {kl:0.6f}")
    var = compute_variance(particles, kl)
    print(f"KL var: {var:0.6f}")
    return (kl, var)

n_reps = 1000
t_length = 20
for ff in sys.argv[1:]:
    run_smc(eval(ff), t_length, n_reps, ff)


# print("Mean KL:", sum(kl_per_token) / len(kl_per_token))
