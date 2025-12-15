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

EOF = len(tokens)-1

max_depth = 4

def normalize(layers, curr):
	layers[len(curr)][*curr] /= np.sum(layers[len(curr)][*curr])

def random_LM(tokens, depth):
	# Transition tables
	layers = {
		i: np.random.rand((len(tokenset),)*(i+1))
		for i in range(depth)
	}

	layers = {0: tt}

	for t in tokens:
		lm = fully_random_LM(tokens, depth-1)



# LMs are randomized decision trees
# A state is simply a string of tokens
class LM:
	def __init__(self, tokenset, depth, seed):
		self.tokenset = tokenset
		# print(self.tokenset)
		self.depth = depth
		np.random.seed(seed)
		self.layers = {
			i: np.random.rand( *(len(tokenset) for _ in range(i+1)))
			for i in range(depth)
		}

	def sample_one(self, state):
		ttable = self.layers[len(state)][*state]
		ttable /= np.sum(ttable)
		# print(np.shape(ttable))
		r = random.random() - ttable[0]
		i = 0
		while(r > 0):
			i += 1
			# print(i)
			r -= ttable[i]
		return i

	def sample_string(self):
		state = []
		while len(state) < self.depth and (len(state) == 0 or state[-1] != EOF):
			state.append(self.sample_one(state))
		if state[-1] != EOF:
			state.append(EOF)
		return state