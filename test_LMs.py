from smoLM import *

A = LM(tokens, max_depth, seed=123456)
B = LM(tokens, max_depth, seed=654321)

# print(A.layers)

for _ in range(5):
	print(to_string(A.sample_string()))
print()
for _ in range(5):
	print(to_string(B.sample_string()))

def MC_KL(A, B, n_reps=100000):
	print(f"Doing Monte Carlo for {n_reps}")
	tot = 0
	for i in range(n_reps):
		sample = A.sample_string()
		p = A.prob(sample)
		q = B.prob(sample)
		if (q == 0):
			tot += float('inf')
		else:
			tot += math.log2(p / q)
		print(f"so far:{tot / (i+1)}\t\t", end='\r')
	print(f"Final estimation: {tot / n_reps}")

KL_AB = calc_KL(A, B)
print(f"KL Divergence between A and B is {KL_AB}")
MC_KL(A, B, 100)
Var_KL_AB = calc_MC_var(A, B, KL_AB)
print(f"The Variance of Monte Carlo is {Var_KL_AB}")
