from smoLM import *

A = LM(tokens, max_depth, seed=123456)

# print(A.layers)

for _ in range(15):
	print(A.sample_string())