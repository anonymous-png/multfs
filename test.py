from dataset_generators import create_directories_and_datasets, create_dir
from extract_class import FeatureScore
from multfs import MultFS
from adhoc_removal import *

from common_utils import gen_csv_from_tuples, read_csv_list, make_query
import sys, time, string, random

from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def millis():
	#print(time.time())
	return int(round(time.time() * 1000))

def random_string(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def combin(n,r):
	return int(np.math.factorial(n) / (np.math.factorial(r) * np.math.factorial(n - r)))

def generate_args_dict():
	args=dict()

	create_dir("test_lin_files")
	create_dir("test_vec_files")

	args['test_lin'] = dict()
	args['test_vec'] = dict()
	

	# Automatic generation of filename for directories created above
	for key in args.keys():
		args[key]['identifier'] = key
		args[key]['dataset_filename'] = key + "_files/user_to_"+ key + "_clean.csv"
		
		args[key]['cdf_filename'] = key + "_files/cdf_"+ key +".pdf"
		
		args[key]['bin_matrix_filename'] = key + "_files/" + key + "_bin_matrix_map.dat"
		args[key]['weight_matrix_filename'] = key + "_files/" + key + "_weight_matrix_map.dat"
		
		args[key]['coincidence_score_filename'] = key + "_files/" + key + "_coin_score.dat"
		args[key]['uniqueness_score_filename'] = key + "_files/" + key + "_uniq_score.dat"
		
		args[key]['coincidence_csv_filename'] = key + "_files/results_coin_" + key + ".csv"
		args[key]['uniqueness_csv_filename'] = key + "_files/results_uniq_"+ key +".csv"
		args[key]['pickle_file'] = key + "_files/.pkl"
		args[key]['rarity_bound'] = None

		args[key]['user_removal'] = None
		args[key]['value_removal'] = None

	return args

def gen_random_dictio_of_users(size_users, size_vals):
	dictio_of_users = {}
	dictio_of_values = {}
	for useri in range(size_users):
		key = random_string(10)
		dictio_of_users[key] = []
	for ipj in range(size_vals):
		key = random_string(10)
		dictio_of_values[key] = []

	lst_users = list(dictio_of_users.keys())
	lst_values = list(dictio_of_values.keys())
	#print(len(dictio_of_users), len(dictio_of_values), size_users, size_vals)
	#print(len(lst_ips), size_vals)
	x = 1
	for i in dictio_of_values.keys():
		num = random.randint(2, int((size_vals - 1) / x))
		for j in range(num):
			order = random.randint(0, size_users - 1)
			#print(order)
			dictio_of_users[lst_users[order]].append(i)
			dictio_of_values[i].append(lst_users[order])

	return dictio_of_users, dictio_of_values

def basic_version(dictio_of_users, dictio_of_values):
	tic = millis()
	fs = FeatureScore("","", "","", "", "", "", "",	"", "", "")
	scores = {}
	dictio_of_weights = fs.gen_dictio_of_weigths(dictio_of_values)
	#dictio_of_users = { k:dictio_of_users[k] for k in list(dictio_of_users.keys())[:n]}
	lst_users = list(dictio_of_users.keys())
	status.create_numbar(100, len(dictio_of_users))
	for ind1, user1 in enumerate(lst_users):
		status.update_numbar(ind1, len(dictio_of_users))
		for ind2, user2 in enumerate(lst_users[ind1:]):
			score_c = 0
			score_u = 1
			for btc1 in dictio_of_users[user1]:
				if btc1 in dictio_of_users[user2]:
					score_u += dictio_of_weights[btc1] ** 2
					#score_c += 1
			scores[user1 + user2] = [score_u, score_c]
	status.end_numbar()
	toc = millis()
	print("basic: tic", tic, "toc", toc, toc-tic)
	return toc - tic

def vectorized_version(args, dictio_of_users, dictio_of_values):
	tic = millis()
	fs = FeatureScore(args['identifier'],args['dataset_filename'], 
		args['cdf_filename'],
		args['bin_matrix_filename'], args['weight_matrix_filename'], 
		args['coincidence_score_filename'], args['uniqueness_score_filename'], 
		args['coincidence_csv_filename'], args['uniqueness_csv_filename'],
		args['rarity_bound'], args['pickle_file'], 
		user_removal=args['user_removal'], value_removal=args['value_removal'])
	print("[+] Generating processing information")
	dictio_of_weights = fs.gen_dictio_of_weigths(dictio_of_values)
	promising_combinations = fs.get_promising_combinations(dictio_of_users, dictio_of_values)
	weight_matrix = fs.gen_weight_matrix_mem(dictio_of_values, dictio_of_users, dictio_of_weights)
	fs.process_pairs_of_users(dictio_of_users, dictio_of_values, promising_combinations, weight_matrix, None)
	toc = millis()
	print("basic: tic", tic, "toc", toc, toc-tic)
	return toc - tic

def execute_performance_tests(args, size_users, size_vals):
	
	dictio_of_users, dictio_of_values = gen_random_dictio_of_users(size_users, size_vals)
	
	time_b = basic_version(dictio_of_users, dictio_of_values)
	time_v = vectorized_version(args['test_vec'], dictio_of_users, dictio_of_values)
	print("Basic version: %d\nVectorized version: %d\n"%(time_b, time_v))
	return time_b, time_v

def execute_user_tests():
	print("executing all tests")
	args = generate_args_dict()
	
	top = 1000 + 1
	lnum = [500, 450, 400]
	for ind, num in enumerate(lnum):
		lst_results = []
		for size_users in range(100,top, 25):
			print("[>>] Size users: %d Size values: %d" %(size_users, num))
			time_b, time_v = execute_performance_tests(args, size_users, num)
			lst_results.append((size_users, num, combin(size_users,2), time_b, time_v))
		gen_csv_from_tuples("user_time_comparison%d.csv" % (ind), ["Users", "Values", "User combinations", "Time Basic", "Time Vector"], lst_results)

def execute_vals_tests():
	print("executing all tests")
	args = generate_args_dict()
	
	top = 1000 + 1
	lnum = [450, 400]
	for ind, num in enumerate(lnum):
		lst_results = []
		for size_vals in range(100,top,50):
			print("[>>] Size users: %d Size values: %d" %(num, size_vals))
			time_b, time_v = execute_performance_tests(args, num, size_vals)
			lst_results.append((num, size_vals, combin(num,2), time_b, time_v))
		gen_csv_from_tuples("value_time_comparison%d.csv" % (ind), ["Users", "Values", "User combinations", "Time Basic", "Time Vector"], lst_results)

def execute_all_tests():
	print("executing all tests")
	#args = generate_args_dict()
	execute_vals_tests()
	#execute_user_tests()
	# top = 1000 + 1
	# num = 500
	# lst_results = []
	# for size_vals in range(100,top,50):
	# 	print("[>>] Size users: %d Size values: %d" %(num, size_vals))
	# 	time_b, time_v = execute_performance_tests(args, num, size_vals)
	# 	lst_results.append((num, size_vals, combin(num,2), time_b, time_v))
	# gen_csv_from_tuples("value_time_comparison.csv", ["Users", "Values", "User combinations", "Time Basic", "Time Vector"], lst_results)
	# num = 450
	# lst_results = []
	# for size_users in range(100,top, 25):
	# 	print("[>>] Size users: %d Size values: %d" %(size_users, num))
	# 	time_b, time_v = execute_performance_tests(args, size_users, num)
	# 	lst_results.append((size_users, num, combin(size_users,2), time_b, time_v))
	# gen_csv_from_tuples("user_time_comparison2.csv", ["Users", "Values", "User combinations", "Time Basic", "Time Vector"], lst_results)
	# lst_results = []
	# num = 400
	# for size_users in range(100,top, 25):
	# 	print("[>>] Size users: %d Size values: %d" %(size_users, num))
	# 	time_b, time_v = execute_performance_tests(args, size_users, num)
	# 	lst_results.append((size_users, num, combin(size_users,2), time_b, time_v))
	# gen_csv_from_tuples("user_time_comparison3.csv", ["Users", "Values", "User combinations", "Time Basic", "Time Vector"], lst_results)

def plot_graph1(U, Y, Y1, Y2, name):

		#print("U", U)
		#print("V", V)
	plt.style.use('seaborn-darkgrid')
	palette = plt.get_cmap('Set1')

	plt.plot(U, Y, color=palette(1), label="V = 500")
	plt.plot(U, Y1, color=palette(3), label="V = 450")
	plt.plot(U, Y2, color=palette(4), label="V = 400")
	#plt.plot(X, U, color=palette(1), label="Users (U) Removal Percentage")
	#plt.plot(X, CY, color=palette(1), linestyle='--', label="CDF(Y)") 
		#marker='o', markersize=3, markerfacecolor=palette(0))
	#plt.axhline(y=1.0, color=palette(2), linestyle='-.', label="Rarity Cutoff")
	#plt.xticks(np.arange(min(X) - 5, max(X)+5, 5.0), rotation='vertical')
	plt.xlabel('Number of Users', fontstyle = 'italic', fontsize=12.0)
	plt.ylabel('Vectorized Time to Basic Time Ratio', fontstyle = 'italic', fontsize=12.0)

	#plt.scatter('', xy=(x, cy), xytext=(0, 0), color='red' , textcoords='offset points') 
	plt.legend()
	plt.savefig(name, format="PDF", bbox_inches='tight')
	plt.clf()

def plot_graph2(V, Y, Y1, Y2, name):

		#print("U", U)
		#print("V", V)
	plt.style.use('seaborn-darkgrid')
	palette = plt.get_cmap('Set1')

	plt.plot(V, Y, color=palette(0), label="U = 500")
	plt.plot(V, Y1, color=palette(5), label="U = 450")
	plt.plot(V, Y2, color=palette(6), label="U = 400")
	#plt.plot(X, U, color=palette(1), label="Users (U) Removal Percentage")
	#plt.plot(X, CY, color=palette(1), linestyle='--', label="CDF(Y)") 
		#marker='o', markersize=3, markerfacecolor=palette(0))
	plt.axhline(y=100.0, color=palette(2), linestyle='-.', label="Cutoff")
	#plt.xticks(np.arange(min(X) - 5, max(X)+5, 5.0), rotation='vertical')
	plt.xlabel('Number of Values', fontstyle = 'italic', fontsize=12.0)
	plt.ylabel('Vectorized Time to Basic Time Ratio', fontstyle = 'italic', fontsize=12.0)

	#plt.scatter('', xy=(x, cy), xytext=(0, 0), color='red' , textcoords='offset points') 
	plt.legend()
	plt.savefig(name, format="PDF", bbox_inches='tight')
	plt.clf()


def plot_graph3(U, V, Y, Y1, Y2, Y3, Y4, Y5, name):

		#print("U", U)
		#print("V", V)
	plt.style.use('seaborn-darkgrid')
	palette = plt.get_cmap('Set1')

	plt.plot(U, Y, color=palette(1), label="V = 500")
	plt.plot(U, Y1, color=palette(3), label="V = 450")
	plt.plot(U, Y2, color=palette(4), label="V = 400")
	plt.plot(V, Y3, color=palette(0), label="U = 500")
	plt.plot(V, Y4, color=palette(5), label="U = 450")
	plt.plot(V, Y5, color=palette(6), label="U = 400")
	#plt.plot(X, U, color=palette(1), label="Users (U) Removal Percentage")
	#plt.plot(X, CY, color=palette(1), linestyle='--', label="CDF(Y)") 
		#marker='o', markersize=3, markerfacecolor=palette(0))
	#plt.axhline(y=100.0, color=palette(2), linestyle='-.', label="Cutoff")
	#plt.xticks(np.arange(min(X) - 5, max(X)+5, 5.0), rotation='vertical')
	plt.xlabel('Values/Users', fontstyle = 'italic', fontsize=12.0)
	plt.ylabel('Speedup', fontstyle = 'italic', fontsize=12.0)

	#plt.scatter('', xy=(x, cy), xytext=(0, 0), color='red' , textcoords='offset points') 
	plt.legend()
	plt.savefig(name, format="PDF", bbox_inches='tight')
	plt.clf()
def graphs_creation():
	lst1 = read_csv_list("user_time_comparison1.csv")[1:]
	lst2 = read_csv_list("user_time_comparison2.csv")[1:]
	lst3 = read_csv_list("user_time_comparison3.csv")[1:]
	U = np.array([int(x[0]) for x in lst1])
	V = np.array([int(x[1]) for x in lst1])
	TB = np.array([int(x[3]) for x in lst1])
	TV = np.array([int(x[4]) for x in lst1])
	TB1 = np.array([int(x[3]) for x in lst2])
	TV1 = np.array([int(x[4]) for x in lst2])
	TB2 = np.array([int(x[3]) for x in lst3])
	TV2 = np.array([int(x[4]) for x in lst3])

	Y = TV / TB * 100
	Y1 = TV1 / TB1 * 100
	Y2 = TV2 / TB2 * 100
	plot_graph1(U, Y, Y1, Y2, "user_to_prop.pdf")
	Y = TB / TV 
	Y1 = TB1 / TV1 
	Y2 = TB2 / TV2 

	lst1 = read_csv_list("value_time_comparison3.csv")[1:]
	lst2 = read_csv_list("value_time_comparison0.csv")[1:]
	lst3 = read_csv_list("value_time_comparison1.csv")[1:]
	U2 = np.array([int(x[0]) for x in lst1])
	V = np.array([int(x[1]) for x in lst1])
	TB = np.array([int(x[3]) for x in lst1])
	TV = np.array([int(x[4]) for x in lst1])
	TB1 = np.array([int(x[3]) for x in lst2])
	TV1 = np.array([int(x[4]) for x in lst2])
	TB2 = np.array([int(x[3]) for x in lst3])
	TV2 = np.array([int(x[4]) for x in lst3])

	Y3 = TV / TB * 100
	Y4 = TV1 / TB1 * 100
	Y5 = TV2 / TB2 * 100
	plot_graph2(V, Y3, Y4, Y5, "value_to_prop.pdf")
	Y3 = TB / TV 
	Y4 = TB1 / TV1 
	Y5 = TB2 / TV2 
	plot_graph3(U, V, Y, Y1, Y2, Y3, Y4, Y5, "speedup_to_prop.pdf")

def linear_reg():
	lst1 = read_csv_list("user_time_comparison1.csv")[1:]
	lst2 = read_csv_list("user_time_comparison2.csv")[1:]
	lst3 = read_csv_list("user_time_comparison3.csv")[1:]
	lst4 = read_csv_list("user_time_comparison1.csv")[1:]
	lst5 = read_csv_list("user_time_comparison2.csv")[1:]
	lst6 = read_csv_list("user_time_comparison3.csv")[1:]
	lst_global = lst1 + lst2 + lst3 + lst4 + lst5 + lst6
	U = np.array([int(x[0]) for x in lst_global]).reshape(len(lst_global),1)
	V = np.array([int(x[1]) for x in lst_global]).reshape(len(lst_global),1)
	TB = np.array([int(x[3]) for x in lst_global]).reshape(len(lst_global),1)
	TV = np.array([int(x[4]) for x in lst_global]).reshape(len(lst_global),1)
	print(U.shape, V.shape)
	X = np.concatenate((U, V), axis=1)
	print(X.shape, TV.shape)
	regv = LinearRegression().fit(X, TV)
	regb = LinearRegression().fit(X, TB)
	return regv, regb
	#print(X[0], TV[0])
	#print(reg.predict(X[0].reshape(1,-1)))

def main():
	if len(sys.argv) < 2:
		print("""Usage: python3 __init__.py <option>
	<option>:
		'performance': do performance test
		'cfs': calculates feature scores
		'multfs': calculates multfs from score files
		'cdf': generates CDFs for features configured
		'evred': generates reduction evaluation csv
		'distcsv': generates the distribution csvs
		'optcsv': generates the optimal reduction csvs
		'filejoin': joins all files into a single one""")
		return

	args = generate_args_dict()

	if sys.argv[1] == 'performance':
		print("[>>] Doing all")
		execute_all_tests()

	elif sys.argv[1] == 'graphs':
		print("[>>] Calculating feature scores")
		graphs_creation()

	elif sys.argv[1] == 'linreg':
		print("[>>] Linear Regression")
		linear_reg()

	else:
		print("""Usage: python3 __init__.py <option>
	<option>:
		'all': do all processing
		'cfs': calculates feature scores
		'multfs': calculates multfs from score files
		'cdf': generates cdfs for features configured
		'evred': generates reduction evaluation csv
		'distcsv': generates the distribution csvs""")

if __name__ == "__main__":
	main()
