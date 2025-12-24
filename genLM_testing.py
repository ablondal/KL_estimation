from genlm.control import PromptedLLM, BoolFSA, AWRS
import asyncio
import hashlib

# Create two language model potentials:
# Qwen/Qwen2.5-0.5B
# Qwen/Qwen2.5-1.5B
q = PromptedLLM.from_name("Qwen/Qwen2.5-0.5B")
p = PromptedLLM.from_name("Qwen/Qwen2.5-1.5B")
p.set_prompt_from_str("Here is an interesting fact:")
q.set_prompt_from_str("Here is an interesting fact:")

# Create a finite-state automaton potential using a regular expression.
fsa = BoolFSA.from_regex(r" SMC is (🔥🔥|😍😍|🤌🤌) with LMs")

# Coerce the FSA so that it operates on the token type of the language model.
coerced_fsa = fsa.coerce(q, f=b"".join)

# Create a token sampler that combines the language model and FSA.
token_sampler = AWRS(q, coerced_fsa)

# Generate text using SMC.
# Generation is asynchronous; use `await` if calling in an async context (like in an async
# function or in a Jupyter notebook) and `asyncio.run(token_sampler.smc(...))` otherwise.
sequences = asyncio.run(token_sampler.smc(
    n_particles=10, # Number of candidate sequences to maintain
    ess_threshold=0.5, # Threshold for resampling
    max_tokens=30, # Maximum sequence length
    verbosity=1 # Print particles at each step
))

print(sequences.decoded_posterior)
# Example output:
# {
#   ' SMC is 🔥🔥 with LMs': 1.0,
# }