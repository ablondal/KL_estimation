import sys, os, json, math
script_directory = os.path.dirname(os.path.realpath(__file__))

methods = {}

files = os.listdir(script_directory)
for file in files:
	if not file.endswith('.json'):
		continue

	name = file[:-5]
	with open(os.path.join(script_directory, file), 'r') as f:
		runs = json.load(f)
	M = len(runs)

	method = {}

	if file == 'RB.json':
		mean = 0
		for run in runs:
			run_kl = sum(run)
			mean += run_kl / M
		# print(f"===== {file[:-5]} =====")
		# print(f"KL estimation: {mean:.4f}")
		var = 0
		for run in runs:
			var += (mean - sum(run))**2 / (M**2)
		# print(f"Sample variance: {var:.4f}")
		method['mean'] = mean
		method['var'] = method['mse'] = var
		method['ess'] = 100
		
	else:
		mean = 0
		mean2 = 0
		tot_w = 0
		for run in runs:
			mean += run["weight"] * run["val"]
			tot_w += run["weight"]
			mean2 += run["weight"] * run["val"]
		mean /= tot_w
		mean2 /= M
		print(f"{name}: {mean:.4f}, {mean2:.4f}, {math.fabs(mean-mean2):.4f}")
		# print(f"===== {file[:-5]} =====")
		# print(f"KL estimation: {mean}")
		mse = 0
		var = 0
		var2 = 0
		essinv = 0
		for run in runs:
			var += ((run["weight"] / tot_w)**2) * (run["val"] - mean)**2
			mse += (run["weight"] / tot_w) * (run["val"] - mean)**2
			essinv += (run["weight"] / tot_w)**2
		method['mean'] = mean
		method['mse'] = mse
		method['var'] = var
		method['ess'] = 100 / (M*essinv)
		# print(f"Sample variance for {name}: {var}")

	methods[name] = method

KL0 = methods['RB']['mean']
var0 = methods['just_p']['var']

for name in methods:
	abs_err = math.fabs(methods[name]['mean'] - KL0)
	methods[name]['abs_err'] = abs_err
	# Combined metric 0.7 × Error/KL0 + 0.3 × Var/σ2 MC (lower is better)
	methods[name]['score'] = (
		0.7 * abs_err / KL0
		+ 0.3 * methods[name]['var'] / var0
	)
	# print(name, methods[name])

##### PRINT FOR USE IN A TABLE #####
for name in methods:
	# if name == 'RB':
	# 	continue
	m = methods[name]
	# \textbf{Baseline} & \textbf{KL estimate} & \textbf{MSE} & \textbf{Variance} & \textbf{ESS (\%)} & \textbf{Score}
	print(f"{name} &{m['abs_err']:0.4f} &{m['mse']:0.4f} &{m['var']:0.4f} &{m['ess']:0.2f} &{m['score']:0.4f}\\\\")

