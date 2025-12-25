import torch, math, json, sys
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    log_p = 0
    log_q = 0

    p_probs = []
    q_probs = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            out_a = model_a(input_ids)
            out_b = model_b(input_ids)

            logits_a = out_a.logits[0, -1] / temperature
            logits_b = out_b.logits[0, -1] / temperature

            l_p = torch.log_softmax(logits_a, dim=-1)
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

def run_smc(func, length, reps, outp):
    print(f"Proposal distribution: {outp}\nReps, Length: {reps}, {length}")

    particles = []
    kl_sum = 0
    w_sum = 0
    for k in range(reps):
        text, lg_r, lg_p, lg_q = sample_once(
            "An interesting fact is that",
            max_new_tokens=length,
            temperature=0.4,
            next_token_weight = func
        )
        weight = math.exp(lg_p - lg_r)

        particles.append(
            {'text': text,
            'weight': weight,
            'val': lg_p - lg_q})

        # print(text)
        # print(f"The weight is {weight:0.6f}")
        # print(f"Value of lg_p - lg_q is {lg_p - lg_q}")
        kl_sum += weight * (lg_p - lg_q)
        w_sum += weight
        print(f"KL: {kl_sum / w_sum}\t\t", end='\r')

    with open('tests2/'+outp+'.json', 'w') as f:
        json.dump(particles, f, indent=2)

    kl = kl_sum / w_sum
    print(f"KL estimate: {kl:0.6f}")
    # calc variance
    var_sum = 0
    for p in particles:
        var_sum += p['weight'] * (p['val'] - kl) * (p['val'] - kl)
    var = var_sum / w_sum
    print(f"KL var: {var:0.6f}")

n_reps = 1000
t_length = 20
for ff in sys.argv[1:]:
    run_smc(eval(ff), t_length, n_reps, ff)


# print("Mean KL:", sum(kl_per_token) / len(kl_per_token))