import psycopg2, json, csv
import re
import os
import status
import operator
import math
import numpy as np
import multiprocessing as mp
import itertools
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from threading import Thread
import threading
import time
import functools
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from random import shuffle
from common_utils import gen_csv_from_tuples, read_csv_list, make_query, file_len, notify_mail

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MultFS(object):

	def __init__(self, combination_filename):
		self.filenames = dict()
		self.filenames['combination_filename'] = combination_filename
		self.list_files = []

	def add_file(self, filename, prefix):
		self.list_files.append((filename, prefix))

	def get_file_lengths(self):
		dictio = {}
		for filename, prefix  in self.list_files:
			dictio[prefix] = file_len(filename)
	
	def get_joined_results(self, filename):
		dictio_of_results = {}
		lst_results = read_csv_list(filename)
		head = lst_results[0]
		lst_results = lst_results[1:]
		for entry in lst_results:
			user0, user1 = entry[0], entry[1]
			for indi, prefix in enumerate(head[2:]):
				if not entry[0] in dictio_of_results.keys():
					dictio_of_results[entry[0]] = dict()
				if not entry[1] in dictio_of_results[entry[0]].keys():
					dictio_of_results[entry[0]][entry[1]] = dict()
				dictio_of_results[entry[0]][entry[1]][prefix] = float(entry[2 + indi])

		return dictio_of_results, lst_results

	def dictio_of_results_to_list(self, dictio_of_results):
		general_list = []
		for user1 in dictio_of_results.keys():
			for user2, values in dictio_of_results[user1].items():
				#if uncommented we take into account only coincidences of values and users
				#if len(values) <= 2:
					#continue
				user_res = [user1, user2]
				for _, prefix in self.list_files:
					if prefix in dictio_of_results[user1][user2]:
						user_res.append(dictio_of_results[user1][user2][prefix])
					else:
						user_res.append(0)
				general_list.append(tuple(user_res))
		print(general_list[0])
		return general_list

	def store_joined_results(self, general_list, filename):
		#general_list = self.dictio_of_results_to_list(dictio_of_results)
		head = ["IdAuthor1", "IdAuthor2"] + [prefix for _, prefix in  self.list_files]
		gen_csv_from_tuples(filename , head, general_list)

	def store_normalized_results(self, users, normalized_matrix, filename):
		head = ["IdAuthor1", "IdAuthor2"] + [prefix for _, prefix in  self.list_files]
		lst = [tuple([pair[0], pair[1]] + [x for x in values]) for pair, values in zip(users, normalized_matrix)]
		gen_csv_from_tuples(filename , head, lst)

	def read_list_with_format(self, filename):
		lst_users = read_csv_list(filename)
		for i in range(len(lst_users)):
			entry = list(lst_users[i])
			for j in range(2, len(entry)):
				entry[j] = float(entry[j])
			lst_users[i] = entry
		return lst_users


	def join_all_results(self):
		dictio_of_results = dict()
		tic = time.time()
		toc = time.time()
		for indi, tup in enumerate(self.list_files):
			filename, prefix = tup[0], tup[1]
			print("[-] Going for file: %d - %s" % (indi, filename))
			
			lst_results = read_csv_list(filename)[1:]
			filelen = len(lst_results)
			print("[+] Sorting list")
			lst_results = sorted(lst_results, key=lambda x: x[0] + x[1], reverse=False)
			status.create_numbar(100, filelen)
			for indj, entry in enumerate(lst_results):
				
				status.update_numbar(indj, filelen)
				if not entry[0] in dictio_of_results.keys():
					dictio_of_results[entry[0]] = dict()
				if not entry[1] in dictio_of_results[entry[0]].keys():
					dictio_of_results[entry[0]][entry[1]] = dict()

				dictio_of_results[entry[0]][entry[1]][prefix] = float(entry[2])

			status.end_numbar()

			print("[+] Ended with file: %d - %s in %d seconds" % (indi, filename, time.time() - tic))
			
		return dictio_of_results

	def order_users(self, entry):
		if entry[0] > entry[1]:
			return entry[1], entry[0]
		else:
			return entry[0], entry[1]

	def join_all_results_alt(self):
		dictio_of_results = dict()
		tic = time.time()
		toc = time.time()
		for indi, tup in enumerate(self.list_files):
			filename, prefix = tup[0], tup[1]
			print("[-] Going for file: %d - %s" % (indi, filename))
			filelen = 102800000
			with open(filename, 'r') as f:
				indj = 0
				status.create_numbar(100, filelen)
				for line in csv.reader(f, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL):
					indj += 1
					if indj == 1:
						#The first line of the csv are titles
						continue
					entry = tuple(line)			
					#lst_results = read_csv_list(filename)[1:]
					user1, user2 = self.order_users(entry)
					status.update_numbar(indj, filelen)
					if '-1' in user1 or '-1' in user2:
						continue

					
					if not user1 in dictio_of_results.keys():
						dictio_of_results[user1] = dict()
					if not user2 in dictio_of_results[user1].keys():
						dictio_of_results[user1][user2] = dict()

					dictio_of_results[user1][user2][prefix] = float(entry[2])

				status.end_numbar()
			notify_mail("[+] Ended with file: %d - %s in %d seconds" % (indi, filename, time.time() - tic))
		return dictio_of_results

	def normalize_data(self,lst_results):
		users = [x[:2] for x in lst_results]
		matrix = np.array([list(x[2:]) for x in lst_results])
		matrix = 1 - (matrix / matrix.max(axis=0)) # Max by columns_:
		return users, matrix

	def multfs_calculation(self, users, matrix):
		num_metrics = len(self.list_files)
		num_features = math.ceil(num_metrics / 2)
		print("Num Features", num_features)
		intervals = np.arange(0.0, 1.0, 1.0/(num_features - 1))[1:]
		# We calculate the mean of the values
		mean = np.mean(matrix, axis = 1)
		std = np.std(matrix, axis = 1)
		mean /= mean.max(axis=0)

		
		nums = [0.0] + list(np.arange(0.0, 1.0, 1.0/(num_features - 1))[1:]) #+ [0.9999999999]
		counts = [np.count_nonzero(matrix <= count, axis = 1) for count in nums]

		ones = np.count_nonzero(matrix < 1.0, axis = 1)
		nums.append(0.999999999999)
		counts.append(ones)
		# We compute the weights of the different part
		#zeros = (num_metrics - np.count_nonzero(matrix == 0, axis = 1)) / float(num_metrics)
		quarters = [(num_metrics - count) / float(num_metrics) for count in counts]
		
		#stdmean = std * mean
		quarters_pond = np.ones(ones.shape)
		for weight, ponderation in enumerate(quarters):
			quarters_pond *= (ponderation + ((1 - std) ** 2 ))
		
		#quarters_pond = ((num_metrics - quarters_pond) / num_metrics)
		metric_matrix = mean * (quarters_pond ** 1) #* std
		#ones = np.count_nonzero(matrix == 1, axis = 1) / float(num_metrics)
		metric_matrix /= metric_matrix.max(axis=0)
		# We declare the results list
		mean = np.around(mean, decimals=3)
		results = []
		
		#ones_c = np.count_nonzero(matrix >= 1.0, axis = 1)
		#quarter1_c = np.count_nonzero(matrix >= 0.75, axis = 1)
		#quarter2_c = np.count_nonzero(matrix >= 0.5, axis = 1)
		#quarter3_c = np.count_nonzero(matrix >= 0.25, axis = 1)
		#quarter4_c = np.count_nonzero(matrix >= 0.01, axis = 1)
		# szeros_c = np.count_nonzero(matrix == 0, axis = 1)
 		#ones_c = np.count_nonzero(matrix == 1, axis = 1) 
		#print(len(users), ones_c.shape)
		## TRY CODE BELOW: Vectorized version of the former
		#metric_matrix = mean * zeros * quarter1 * quarter2 * quarter3 * quarter4 * ones
		#metric_matrix = mean
		#print(users[0], matrix[0], ones_c[0], quarter1_c[0])
		#metric_matrix2 = mean * zeros * quarter1 * quarter2 * quarter3 * quarter4
		res_list = []
		counts = np.array(counts).T
		print(counts.shape, metric_matrix.shape)
		for pair, metric, mu, count in zip(users, metric_matrix, mean, counts):
			pair_scores = [pair[0], pair[1],  metric, mu] + count.tolist()
			#print(count)
			res_list.append(tuple(pair_scores))
		print(nums)
		names = ["user_a", "user_b", "metric", "mean"] + ["<=" + str(num) for num in nums]
		print(names)
		#res_list = [(pair[0], pair[1],  metric, meanx, one, q1, q2, q3, q4, zero) 
		#for pair, metric, meanx, zero, q1, q2, q3, q4, one 
		#in zip(users, metric_matrix, mean, zeros_c , quarter1_c, quarter2_c, quarter3_c, quarter4_c, ones_c) ]
		
		# res_list = []
		# for pair, metric, zero, q1, q2, q3, q4, one  in zip(users, metric_matrix, ones_c, quarter1_c, quarter2_c, quarter3_c, quarter4_c, zeros_c):
		# 	res_list.append((pair[0], pair[1],  metric, one, q1, q2, q3, q4, zero))
		# for indi, users in enumerate(users):
		# 	user1, user2 = users[0], users[1]
		# 	metric = mean[indi] * zeros[indi] * quarter1[indi] * quarter2[indi] * quarter3[indi] * quarter4[indi] * ones[indi]
		# 	a = (i[0], i[1], metric, mean_arr[indi], np.rint((1 - zeros[indi]) * 10), 
		# 		np.rint((1 - quarter1[indi]) * 10), np.rint((1 - quarter2[indi]) * 10), 
		# 		np.rint((1 - quarter3[indi]) * 10), np.rint((1 - quarter4[indi]) * 10), 
		# 		np.rint(ones[indi] * 10))
		# 	results.append(a)

		return res_list, metric_matrix, names

	def generate_graph(self, filename = "graph.gexf", limit = 1000000000):
		print("[-] Extracting data")
		lst = read_csv_list("weighted_average.csv")[1:]
		print("[-] Generating list")
		#from_nodes = [x[0] for x in lst]
		#to_nodes = [x[1] for x in lst]
		#weight = [x[2] for x in lst]
		elist = [(x[0], x[1], x[2]) for x in lst if float(x[2]) <= limit]
		print("[-] Generating graph")
		G = nx.Graph()
		G.add_weighted_edges_from(elist)
		#print("[-] Pickling")
		#nx.write_gexf(G, "graph.gexf")
		#nx.write_gpickle(G, "graph.pickle")
		return G

	def analyze_connected_components(self):
		G = self.generate_graph()
		for i, graph in enumerate(list(nx.connected_component_subgraphs(G))):
			num_nodes = graph.number_of_nodes()
			print("[-] Going for %d with %d" %(i,num_nodes))
			if num_nodes > 7:
				graph_lst = []
				for user, data in graph.nodes(data=True):
					graph_lst.append((user, graph.degree(user)))
				graph_lst = sorted(graph_lst, key=lambda x: x[1], reverse=True)
				gen_csv_from_tuples("graphs_info/%d-%d.csv"%(num_nodes,i), ["User", "#"], graph_lst)

	def generate_connected_components(self):
		G = self.generate_graph()
		print("[-] Computing connected components")
		connected_components = list(nx.connected_component_subgraphs(G))
		dic_conn = {}
		lst_temp = []
		print("[-] Extracting info from components")
		for i, graph in enumerate(connected_components):
			#lst_temp = [i, graph.number_of_nodes()]
			num_nodes = graph.number_of_nodes()
			if not num_nodes in  dic_conn:
				dic_conn[num_nodes] = 0
			dic_conn[num_nodes] += 1
		lst_results = [(k,v) for k,v in dic_conn.items()]
		print("[-] Sorting")
		lst_results = sorted(lst_results, key=lambda x: x[1], reverse=True)
		gen_csv_from_tuples("graph_connections.csv", ["NUM NODES GRAPH", "#"], lst_results)
	def generate_graph_pickle():
		lst = read_csv_list("weighted_average.csv")[1:]

	def do_combinations(self, filename=None):
		combined_results = "combined_results.csv"
		if filename is None:
			filename = self.filenames['combination_filename'] if os.path.isfile(self.filenames['combination_filename']) else None
		dictio_of_results = None
		list_of_results = None
		if not filename is None:
			dictio_of_results, _ = self.get_joined_results(filename)
			list_of_results = self.dictio_of_results_to_list(dictio_of_results)
			#self.store_joined_results(list_of_results, combined_results)
		else:
			dictio_of_results = self.join_all_results_alt()
			list_of_results = self.dictio_of_results_to_list(dictio_of_results)
			self.store_joined_results(list_of_results, combined_results)

		#list_of_results = self.dictio_of_results_to_list(dictio_of_results)
		#self.store_joined_results(list_of_results, combined_results)
		users, normalized_matrix = self.normalize_data(list_of_results)
		self.store_normalized_results(users, normalized_matrix, "normalized_combined_results.csv")


	def do_all(self, filename=None):
		combined_results = "combined_results.csv"
		if filename is None:
			filename = self.filenames['combination_filename'] if os.path.isfile(self.filenames['combination_filename']) else None
		dictio_of_results = None
		list_of_results = None
		if not filename is None:
			dictio_of_results, _ = self.get_joined_results(filename)
			list_of_results = self.dictio_of_results_to_list(dictio_of_results)
			#self.store_joined_results(list_of_results, combined_results)
		else:
			dictio_of_results = self.join_all_results_alt()
			list_of_results = self.dictio_of_results_to_list(dictio_of_results)
			self.store_joined_results(list_of_results, combined_results)

		#list_of_results = self.dictio_of_results_to_list(dictio_of_results)
		#self.store_joined_results(list_of_results, combined_results)
		users, normalized_matrix = self.normalize_data(list_of_results)
		self.store_normalized_results(users, normalized_matrix, "normalized_combined_results.csv")
		res_list, metric_matrix, names = self.multfs_calculation(users, normalized_matrix)
		res_list = sorted(res_list, key=lambda x: str(x[2]) + str(x[3]) + x[0] + x[1], reverse=False)
		gen_csv_from_tuples("weighted_average.csv", 
			names,
			res_list)
		res_list = sorted(res_list, key=lambda x: str(x[3]) + str(x[2]) + x[0] + x[1], reverse=False)
		gen_csv_from_tuples("weighted_average_1.csv", 
			names,
			res_list)

