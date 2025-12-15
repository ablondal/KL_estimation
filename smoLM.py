import numpy as np
import math
import random

#########################################
#										#
#	A small library for small toy LMs	#
#										#
#########################################

tokens = [chr(ord('a') + a) for a in range(26)]\
	+ [chr(ord('A') + a) for a in range(26)]\
	+ [chr(ord('0') + a) for a in range(10)]\
	+ ['.'] # Functions as EOF

def to_string(arr):
	return ''.join([tokens[a] for a in arr])

EOF = len(tokens)-1

max_depth = 4

# LMs are randomized decision trees
# A state is simply a string of tokens
class LM:
	# Initialize a random LM with the given seed
	def __init__(self, tokenset, depth, seed):
		self.tokenset = tokenset
		# print(self.tokenset)
		self.depth = depth
		np.random.seed(seed)
		self.layers = {
			i: np.random.rand( *(len(tokenset) for _ in range(i+1)))
			for i in range(depth)
		}
		self.fix_probs()

	# Probabilities are set randomly, must be normalized to sum to 1
	def fix_probs(self, curr=[]):
		self.layers[len(curr)][*curr] /= np.sum(self.layers[len(curr)][*curr])
		if len(curr) < max_depth-1:
			for i in range(len(self.tokenset)-1): # Excluding EOF
				self.fix_probs(curr + [i])

	# Samples a next token from the LM, given the state
	def sample_one(self, state):
		ttable = self.layers[len(state)][*state]
		r = random.random() - ttable[0]
		i = 0
		while(r > 0):
			i += 1
			r -= ttable[i]
		return i

	# Samples until EOF (or depth limit)
	def sample_string(self):
		state = []
		while len(state) < self.depth and (len(state) == 0 or state[-1] != EOF):
			state.append(self.sample_one(state))
		if state[-1] != EOF:
			state.append(EOF)
		return state

	# Given a string, calculates the probability that the LM
	# 	would output it
	def prob(self, s):
		tot = 1
		for i in range(min(len(s)-1, self.depth)):
			tot *= self.layers[i][*s[:i+1]]
		return tot

# Calculates the KL divergence between two LMs
# Rigorous and exact
def calc_KL(p: LM, q: LM, depth=max_depth):
	def recur(curr=[], p_prob=1, q_prob=1):
		if len(curr) == depth or (len(curr) > 0 and curr[-1] == EOF):
			if q_prob == 0:
				return float('inf')
			return p_prob * (math.log2(p_prob / q_prob))
		total = 0

		ptable = p.layers[len(curr)][*curr]
		qtable = q.layers[len(curr)][*curr]
		for i in range(len(tokens)):
			total += recur(curr + [i], p_prob*ptable[i], q_prob*qtable[i])
		return total
	return recur()

# Calculates the variance of the Monte Carlo estimation method
# = sum_x (p(x) * [(log p(x) / q(x)) - mu]^2)
# where mu is the actual KL divergence
def calc_MC_var(p: LM, q: LM, mu, depth=max_depth):
	def recur(curr=[], p_prob=1, q_prob=1):
		if len(curr) == depth or (len(curr) > 0 and curr[-1] == EOF):
			if q_prob == 0:
				return float('inf')
			return p_prob * (math.log2(p_prob / q_prob) - mu)**2
		total = 0

		ptable = p.layers[len(curr)][*curr]
		qtable = q.layers[len(curr)][*curr]
		for i in range(len(tokens)):
			total += recur(curr + [i], p_prob*ptable[i], q_prob*qtable[i])
		return total
	return recur()
