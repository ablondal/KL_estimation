import sys, os, json
script_directory = os.path.dirname(os.path.realpath(__file__))

files = os.listdir(script_directory)
for file in files:
	if not file.endswith('.json'):
		continue

	with open(os.path.join(script_directory, file), 'r') as f:
		runs = json.load(f)
	M = len(runs)

	if file == 'RB.json':
		mean = 0
		for run in runs:
			run_kl = sum(run)
			mean += run_kl / M
		print(f"===== {file[:-5]} =====")
		print(f"KL estimation: {mean}")
		var = 0
		for run in runs:
			var += (mean - sum(run))**2 / M
		print(f"Sample variance: {var}")
		
	else:
		mean = 0
		tot_w = 0
		for run in runs:
			mean += run["weight"] * run["val"]
			tot_w += run["weight"]
		mean /= tot_w
		print(f"===== {file[:-5]} =====")
		print(f"KL estimation: {mean}")
		var = 0
		for run in runs:
			var += M * ((run["weight"] / tot_w)**2) * (run["val"] - mean)**2
		print(f"Sample variance: {var}")

