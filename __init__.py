from dataset_generators import create_directories_and_datasets, create_dir, create_directories_and_datasets_1, create_directories_and_datasets_2, generate_directories_for_users
from extract_class import FeatureScore
from multfs import MultFS
from adhoc_removal import *

from common_utils import gen_csv_from_tuples, read_csv_list, make_query
import sys, time, os
import multiprocessing as mp

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from test import linear_reg

def generate_args_dict():
	args=dict()

	
	args['skype'] = dict()
	args['email'] = dict()
	args['btc'] = dict()
	
	args['ip'] = dict()
	args['link'] = dict()

	
	

	# Automatic generation of filename for directories created above
	for key in args.keys():
		args[key]['identifier'] = key
		args[key]['dataset_filename'] = key + "_files/user_to_"+ key + ".csv"

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

	args['ip']['value_removal'] = [ip_value_removal]
	args['link']['value_removal'] = [link_value_removal_keep_params, link_value_removal_2]
	#args['link']['user_removal'] = [link_user_removal]
	args['ip']['user_removal'] = [ip_user_removal]
	args['email']['user_removal'] = [email_user_removal]
	args['skype']['user_removal'] = [skype_user_removal]

	args['skype']['rarity_bound'] = 4
	args['email']['rarity_bound'] = 0
	args['link']['rarity_bound'] = 15
	args['ip']['rarity_bound'] = 23
	args['btc']['rarity_bound'] = 15

	return args

def feature_extraction():
	create_directories_and_datasets()

def generate_feature_scores(args):
	
	lst_fs = []
	for key in args.keys():
		#print("[-] Starting with: %s" %(key))
		fs = FeatureScore(args[key]['identifier'],args[key]['dataset_filename'], 
			args[key]['cdf_filename'],
			args[key]['bin_matrix_filename'], args[key]['weight_matrix_filename'], 
			args[key]['coincidence_score_filename'], args[key]['uniqueness_score_filename'], 
			args[key]['coincidence_csv_filename'], args[key]['uniqueness_csv_filename'],
			args[key]['rarity_bound'], args[key]['pickle_file'], 
			user_removal=args[key]['user_removal'], value_removal=args[key]['value_removal'])
		
		lst_fs.append(fs)
	return lst_fs

def calculate_feature_scores(lst_fs):
	lst_sparse = ["ip", "link"]
	for fs in lst_fs:
		tic = time.time()
		print("[-] Starting with: %s" %( fs.identifier() ))
		if fs.identifier() in lst_sparse:
			fs.do_all_sparse_matrix_link()
		else:
			fs.do_all()
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))


def calculate_feature_score(lst_fs, feature_name, sparse=False):
	for fs in lst_fs:
		if fs.identifier() == feature_name:
			tic = time.time()
			print("[-] Starting with: %s" %( fs.identifier() ))
			if sparse:
				fs.do_all_sparse_matrix_link()
			else:
				fs.do_all()

			#Generate optimal reduction csv and graphs
			fs.filenames['dataset_filename'] = fs.identifier() + "_files/user_to_"+ fs.identifier() + "_clean.csv"
			fs.gen_optimal_reduction(fs.identifier() + "_files/" + fs.identifier() + "_optimal_dist.csv",
				fs.identifier() + "_files/" + fs.identifier() + "_optimal_dist.pdf" )

			print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))
def calculate_feature_scores_notdont(lst_fs, dont):
	lst_sparse = ["ip", "link"]
	for fs in lst_fs:

		tic = time.time()
		print("[-] Starting with: %s" %( fs.identifier() ))
		if fs.identifier() in dont:
			continue
		if fs.identifier() in lst_sparse:
			fs.do_all_sparse_matrix()
		else:
			fs.do_all()
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))

def calculate_feature_scores_dont(lst_fs, dont):
	for fs in lst_fs:

		tic = time.time()
		
		if not fs.identifier() in dont:
			continue
		print("[-] Starting with: %s" %( fs.identifier() ))
		fs.do_all_sparse_matrix_link()
					#Generate optimal reduction csv and graphs
		fs.filenames['dataset_filename'] = fs.identifier() + "_files/user_to_"+ fs.identifier() + "_clean.csv"
		fs.gen_optimal_reduction(fs.identifier() + "_files/" + fs.identifier() + "_optimal_dist.csv",
			fs.identifier() + "_files/" + fs.identifier() + "_optimal_dist.pdf" )
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))


def caculate_combinations(args, dont, filename=None):
	multfs = MultFS("combined_results.csv")
	for key in args.keys():
		if not (key in dont):
			multfs.add_file(args[key]['coincidence_csv_filename'], key + "coin")
			multfs.add_file(args[key]['uniqueness_csv_filename'], key + "uniq")
	multfs.do_combinations(filename)


def calculate_multfs_score(args, filename=None):
	multfs = MultFS("combined_results.csv")
	for key in args.keys():
		multfs.add_file(args[key]['coincidence_csv_filename'], key + "coin")
		multfs.add_file(args[key]['uniqueness_csv_filename'], key + "uniq")
	multfs.do_all(filename)

def compute_graph_combs(args):
	multfs = MultFS("combined_results.csv")
	multfs.generate_connected_components()

def compute_graph_analysis(args):
	multfs = MultFS("combined_results.csv")
	multfs.analyze_connected_components()

def evaluate_reduction(lst_fs):
	lst_red = []
	for fs in lst_fs:
		tic = time.time()
		print("[-] Starting with: %s" %(fs.identifier()))
		ulen1, ulen2, vlen1, vlen2, umean, ustd, vmean, vstd = fs.evaluate_reduction()
		lst_red.append((key, ulen1, ulen2, vlen1, vlen2, umean, ustd, vmean, vstd))
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))

	gen_csv_from_tuples("reduction_evaluation.csv", ["key", "ulen1", "ulen2", "vlen1", "vlen2", "umean",
													 "ustd", "vmean", "vstd"], lst_red)

def generate_cdfs(lst_fs):
	for fs in lst_fs:
		tic = time.time()
		print("[-] Starting with: %s" %(fs.identifier()))
		fs.cdf()
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))

def gen_dist_csv(lst_fs):
	lst_red = []
	for fs in lst_fs:
		tic = time.time()
		print("[-] Starting with: %s" %(fs.identifier()))
		ulen1, ulen2, vlen1, vlen2, umean, ustd, vmean, vstd = fs.evaluate_reduction()
		lst_red.append((key, ulen1, ulen2, vlen1, vlen2, umean, ustd, vmean, vstd))
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))

def gen_optimal_reduction(lst_fs):
	lst_red = []
	for fs in lst_fs:
		tic = time.time()
		print("[-] Starting with: %s" %(fs.identifier()))
		fs.gen_optimal_reduction(fs.identifier() + "_files/" + fs.identifier() + "_optimal_dist.csv",
			fs.identifier() + "_files/" + fs.identifier() + "_optimal_dist.pdf" )
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))

def gen_optimal_reduction_graph(lst_fs):
	lst_vals = []
	for fs in lst_fs:
		tic = time.time()
		print("[-] Starting with: %s" %(fs.identifier()))
		filename = fs.identifier() + "_files/" + fs.identifier() + "_optimal_dist.csv"
		lst_vals.append(fs.get_optimal_reduction_graph(filename))
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))
	
	plt.style.use('seaborn-darkgrid')
	palette = plt.get_cmap('Set1')

	fig, axs = plt.subplots(2, 3, figsize=(15,8))
	#print(axs)
	plt.subplots_adjust(top=0.9)
	fig.suptitle('Rarity Dataset Cleanup', fontsize=16.0)#, fontweight='bold')
	ind = 0
	for axa in axs:
		for ax in axa:
			if ind >= len(lst_vals):
				ax.set_visible(False)
				continue
			elem, fs = lst_vals[ind], lst_fs[ind]
			X, V, U, rarity_bound = elem[0], elem[1], elem[2], elem[3]
			
			if ind == 0:
				ax.plot(X, V, color=palette(0), label="Values (V) Removal Percentage")
				ax.plot(X, U, color=palette(1), label="Users (U) Removal Percentage")

				ax.axvline(x=rarity_bound, color=palette(2), linestyle='-.', label="Rarity Cutoff")
				#ax.xticks(np.arange(min(X), max(X)+5, 5.0), rotation='vertical')
			else: 
				ax.plot(X, V, color=palette(0))
				ax.plot(X, U, color=palette(1))

				ax.axvline(x=rarity_bound, color=palette(2), linestyle='-.')


			ax.set_title("%s" % (fs.identifier().upper()))
			
			ind += 1
			#ax.legend()
			#ax.xlabel('Rarity')
			#ax.ylabel('Percentage of Removal')
	
	#for ax in axs:
		#ax.set_xticks(list(ax.get_xticks()) + [rarity_bound])
	
	fig.legend(loc='best')
	first = True
	for ax in axs.flat:
		ax.set_ylabel('Remaining Percentage', fontstyle = 'italic', fontsize=12.0) # Y label
		ax.set_xlabel('Rarity Removal', fontstyle = 'italic', fontsize=12.0) # X label
		#ax.set(xlabel='Rarity Removal', ylabel='Remaining Percentage')
	#for ax in axs.flat:
		#ax.label_outer()
	fig.savefig("graph_fig.png", format="PNG", bbox_inches='tight')
	fig.savefig("graph_fig.pdf", format="PDF", bbox_inches='tight')
def join_files(lst_fs):
	lst_sparse = ["link", "ip"]
	for fs in lst_fs:
		if fs.identifier() in lst_sparse:
			fs.join_files()
def get_username(user):
	pos_bracket = user.find("[")
	user_id = int(user[:pos_bracket])
	site_id = int(user[pos_bracket + 1:-1])
	#print("[- -] Extracting from DB")
	query = """SELECT "Username", "RegistrationDate", "LastPostDate" FROM "Member" WHERE "IdMember" = %d AND "Site" = %d;""" % (user_id, site_id)
	rows = make_query(query)
	username = rows[0][0]
	regdate = rows[0][1]
	lastpostdate = rows[0][2]
	return username, regdate, lastpostdate

def get_user_posts(user):
	pos_bracket = user.find("[")
	user_id = int(user[:pos_bracket])
	site_id = int(user[pos_bracket + 1:-1])
	#print("[- -] Extracting from DB")
	query = """SELECT "IdPost", "Content" FROM "Post" WHERE ("Site" != 4 OR "CitedPost"[1] = -1) AND "Author" = %d AND "Site" = %d;""" % (user_id, site_id)
	rows = make_query(query)
	rows = [(row[0], row[1]) for row in rows]
	return rows
	#print("[+ +] Done extracting from DB")

def gen_post_coincidences(user1, user2, coincidences):
	user1_posts = get_user_posts(user1)
	user2_posts = get_user_posts(user2)
	directory = "coincidences3/" + user1 + "-" + user2 + "/"
	create_dir(directory)
	for elem in coincidences:
		directory_elem = elem.replace("/", "").replace(":", "").replace("=", "").replace("\\", "").replace("?", "")[:150]
		directory2 = directory + directory_elem + "/"
		create_dir(directory2)
		isin1 = False
		isin2 = False
		file1= directory2 + user1 + ".txt"
		with open(file1, 'w+') as file:
			for post1 in user1_posts:
				if elem in post1[1].lower():
					isin1 = True
					file.write("\n#BEGINPOST: ID[%d]#\n"%(post1[0]))
					file.write(post1[1])
					file.write("\n#ENDPOST: ID[%d]#\n"%(post1[0]))

		file2 = directory2 + user2 + ".txt"
		with open(file2, 'w+') as file: 
			for post2 in user2_posts:
				if elem in post2[1].lower():
					isin2 = True
					file.write("\n#BEGINPOST: ID[%d]#\n"%(post2[0]))
					file.write(post2[1])
					file.write("\n#ENDPOST: ID[%d]#\n"%(post2[0]))
		if not isin1 or not isin2:
			print("%s elem not in %s [%r] or %s [%r]" %(elem, user1, isin1, user2, isin2))
			os.remove(file1)
			os.remove(file2)
			os.rmdir(directory2)
	return directory

def gen_post_coincidences2(index, user1, user2, coincidences):
	user1_posts = get_user_posts(user1)
	user2_posts = get_user_posts(user2)

	directory = "coincidences2/" +str(index)+ "-" +user1 + "-" + user2 + "/"
	create_dir(directory)
	for elem in coincidences:
		directory_elem = elem.replace("/", "").replace(":", "").replace("=", "").replace("\\", "").replace("?", "")[:150]
		directory2 = directory + directory_elem + "/"

		create_dir(directory2)
		isin1 = False
		isin2 = False
		file1= directory2 + user1 + ".txt"
		with open(file1, 'w+') as file:
			for post1 in user1_posts:
				if elem in post1[1]:
					isin1 = True
					file.write("\n#BEGINPOST: ID[%d]#\n"%(post1[0]))
					file.write(post1[1])
					file.write("\n#ENDPOST: ID[%d]#\n"%(post1[0]))

		file2 = directory2 + user2 + ".txt"
		with open(file2, 'w+') as file: 
			for post2 in user2_posts:
				if elem in post2[1]:
					isin2 = True
					file.write("\n#BEGINPOST: ID[%d]#\n"%(post2[0]))
					file.write(post2[1])
					file.write("\n#ENDPOST: ID[%d]#\n"%(post2[0]))
		if not isin1 or not isin2:
			print("%s elem not in %s or %s" %(elem, user1, user2))
			os.remove(file1)
			os.remove(file2)
			os.rmdir(directory2)
		

	return directory

def make_summary_file(directory, string):
	file2 = directory + "000-SUMMARY.txt"
	print(file2)
	with open(file2, 'w+') as file: 
		file.write(string)

def generate_upost_coincidences_dir(lst_fs):
	lst_users = read_csv_list("user_coincidences.csv")
	lst_users = lst_users[2:]
	create_dir("coincidences3/")
	comb_res = read_csv_list("normalized_combined_results.csv")
	head = comb_res[0]
	comb_res = comb_res[1:]
	for index, entry in enumerate(lst_users):
		if entry[0] == "36245[3]" and entry[1] == "36246[3]":
			continue
		directory = gen_post_coincidences(entry[0], entry[1], entry[3:])
		u1, rg1, lp1 = get_username(entry[0])
		u2, rg2, lp2 = get_username(entry[1])
		ind = [x for x,j in enumerate(comb_res) if j[0] == entry[0] and j[1] == entry[1]]
		coins = "\n" + str(comb_res[ind[0]])
		make_summary_file(directory, "[%s]-[%s]-[%s]\n[%s]-[%s]-[%s]\n%s\n%s"%(u1, rg1, lp1, u2, rg2, lp2, str(head), coins))

def generate_upost_coincidences_dir2(lst_fs):
	lst_users = read_csv_list("user_coincidences2.csv")[1:]
	create_dir("coincidences2/")
	for index, entry in enumerate(lst_users):
		gen_post_coincidences2(index, entry[0], entry[1], entry[2:])

def get_user_coincidences(lst_fs):
	
	lst_users = read_csv_list("weighted_average.csv")[1:]

	#lst_users = [(x[0], x[1] for x in lst_users if float(x[2]) < 0.35)
	indi = 0
	for entry in lst_users:
		indi += 1
		if float(entry[2]) >= 0.35:
			break
	lst_users = lst_users[:indi]

	dictio = None
	for fs in lst_fs:
		tic = time.time()
		print("[-] Starting with: %s" %( fs.identifier() ))
		dictio = fs.get_coincidences_for_users(lst_users, dictio)
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))

	res_list = [[key[0], key[1], key[2]] + val for key, val in dictio.items()]
	res_list = sorted(res_list, key=lambda x: x[2])
	gen_csv_from_tuples("user_coincidences.csv", 
			['User1', 'User2', 'Coincidences'], res_list)


def get_user_coincidences2(lst_fs):
	
	lst_users = read_csv_list("rankings_multfs_dgg.csv")[1:]
	interesting = [9,11,15,38,41,42,47,48,49,51,52,53,54,55,56,63,68,71,73,74,75,76,79,81,88,93,97,110,115,116]
	#lst_users = [(x[0], x[1] for x in lst_users if float(x[2]) < 0.35)
	lst_users = [x for ind, x in enumerate(lst_users) if ind in interesting]
	print(len(interesting), len(lst_users))
	dictio = None
	for fs in lst_fs:
		tic = time.time()
		print("[-] Starting with: %s" %( fs.identifier() ))
		dictio = fs.get_coincidences_for_users(lst_users, dictio)
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))

	res_list = [[key[0], key[1]] + val for key, val in dictio.items()]
	gen_csv_from_tuples("user_coincidences2.csv", 
			['User1', 'User2', 'Coincidences'], res_list)

def get_coincidences(lst_fs):
	get_user_coincidences(lst_fs)
	generate_upost_coincidences_dir(lst_fs)

def sizes(lst_fs):
	regv, regb = linear_reg()
	lst = []
	for fs in lst_fs:
		tic = time.time()
		print("[-] Starting with: %s" %(fs.identifier()))
		v = fs.get_sizes()
		ms1, ms2 = regv.predict(v)[0][0], regb.predict(v)[0][0]
		print("BASIC:", ms2 / 1000, "VECTOR", ms1 / 1000)
		print("SPEEDUP: ", ms2/ms1)
		lst.append((fs.identifier(), v[0][0], v[0][1], np.around(ms1 /1000 /60 /60, decimals=2), np.around(ms2 /1000 /60 /60, decimals=2)))
		print("[+] Finished with: %s in %d seconds" %(fs.identifier(), time.time() - tic))

	for i in lst:
		print("%s & %d & %d & %0.2f & %0.2f \\\\" % (i[0], i[1], i[2], i[3], i[4])) 
def main():
	if len(sys.argv) < 2:
		print("""Usage: python3 __init__.py <option>
	<option>:
		'all': do all processing
		'cfs': calculates feature scores
		'multfs': calculates multfs from score files
		'cdf': generates CDFs for features configured
		'evred': generates reduction evaluation csv
		'distcsv': generates the distribution csvs
		'optcsv': generates the optimal reduction csvs
		'filejoin': joins all files into a single one""")
		return

	args = generate_args_dict()

	if sys.argv[1] == 'all':
		print("[>>] Doing all")
		lst_fs = generate_feature_scores(args)
		calculate_feature_scores(lst_fs)
		calculate_multfs_score(args)

	elif sys.argv[1] == 'cfs':
		print("[>>] Calculating feature scores")
		lst_fs = generate_feature_scores(args)
		calculate_feature_scores(lst_fs)

	elif sys.argv[1] == 'multfs':
		filename = None if len(sys.argv) < 3 else sys.argv[2]
		print("[>>] Calculating MultFS")
		calculate_multfs_score(args, filename)

	elif sys.argv[1] == 'cdf':
		print("[>>] Computing Cummulative Distribution Functions")
		lst_fs = generate_feature_scores(args)
		generate_cdfs(lst_fs)

	elif sys.argv[1] == 'evred':
		print("[>>] Computing Reduction Evaluation csv")
		lst_fs = generate_feature_scores(args)
		evaluate_reduction(lst_fs)

	elif sys.argv[1] == 'distcsv':
		print("[>>] Generating Distribution csv")
		lst_fs = generate_feature_scores(args)
		gen_dist_csv(lst_fs)

	elif sys.argv[1] == 'optcsv':

		print("[>>] Generating Optimal Reduction csv")
		for key in args.keys():
			args[key]['dataset_filename'] = key + "_files/user_to_"+ key + "_clean.csv"
		lst_fs = generate_feature_scores(args)
		
		gen_optimal_reduction(lst_fs)

	elif sys.argv[1] == 'optgraph':
		print("[>>] Generating Optimal Reduction Graph")
		lst_fs = generate_feature_scores(args)
		gen_optimal_reduction_graph(lst_fs)

	elif sys.argv[1] == 'filejoin':
		print("[>>] Doing proof")
		lst_fs = generate_feature_scores(args)
		join_files(lst_fs)

	elif sys.argv[1] == 'usercoin':
		print("[>>] Doing proof")
		lst_fs = generate_feature_scores(args)
		get_user_coincidences(lst_fs)

	elif sys.argv[1] == 'usercoin2':
		print("[>>] Doing proof")
		lst_fs = generate_feature_scores(args)
		get_user_coincidences2(lst_fs)

	elif sys.argv[1] == 'postcoin':
		print("[>>] Doing proof")
		lst_fs = generate_feature_scores(args)
		generate_upost_coincidences_dir(lst_fs)

	elif sys.argv[1] == 'postcoin2':
		print("[>>] Doing proof")
		lst_fs = generate_feature_scores(args)
		generate_upost_coincidences_dir2(lst_fs)

	elif sys.argv[1] == 'graphcomp':
		print("[>>] Doing graph composition")
		compute_graph_combs(args)

	elif sys.argv[1] == 'graphanalysis':
		print("[>>] Doing graph composition")
		compute_graph_analysis(args)

	elif sys.argv[1] == 'alternative':
		print("[>>] Creating datasets for all except link.")
		
		#create_directories_and_datasets_1()

		dont = [('link', True)]
		do = [('skype', False), ('email', False), ('btc', False), ('ip', True)]

		filename = None if len(sys.argv) < 3 else sys.argv[2]
		lst_fs = generate_feature_scores(args)
		#calculate except link

		# lst_ps = []
		# print("[>>] Creating dataset for link")
		# #pskype = mp.Process(target=create_directories_and_datasets_2)
		# #pskype.start()

		# print("[>>] Computing the rest of feature scores")
		# for feature in do:
		# 	p = mp.Process(target=calculate_feature_score, args=(lst_fs, feature[0], feature[1]))
		# 	p.start()
		# 	lst_ps.append(p)

		# for p in lst_ps:
		# 	p.join()

		#Generate combinations only
		caculate_combinations(args, [x[0] for x in dont], filename)
		
		#pskype.join()
		calculate_feature_scores_dont(lst_fs, [x[0] for x in dont])

		os.remove("combined_results.csv")
		os.remove("normalized_combined_results.csv")
		
		calculate_multfs_score(args, filename)
		final_process_1 = mp.Process(target=get_coincidences, args=(lst_fs,))
		final_process_1.start()
		final_process_2 = mp.Process(target=generate_directories_for_users)
		final_process_2.start()

		final_process_1.join()
		final_process_2.join()

	elif sys.argv[1] == 'sizes':
		lst_fs = generate_feature_scores(args)
		sizes(lst_fs)
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
