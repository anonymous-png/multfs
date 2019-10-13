import psycopg2, json, csv
import re
import os
import status
import operator
import math
import numpy as np
import multiprocessing
import itertools
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from threading import Thread
import threading
import time
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from random import shuffle
from db_credentials import  *
import requests

##############################################################################################
# FUNCTION: gen_csv_from_tuples
# DESCRIPTION:  Generates a csv with all the links and the user who posted it.
# OUTPUT_FORMAT: (index, "AuthorId", "AuthorName", "Link")
##############################################################################################
def gen_csv_from_tuples(name, titles, rows):
	#file = open('id_user_url.csv', mode='w+')
	file = open(name, mode='w+')
	writer = csv.writer(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
	writer.writerow(titles)
	for row in rows:
		writer.writerow(row)
##############################################################################################
# FUNCTION: read_csv_as_list
# DESCRIPTION:  Generates a list of tuples of the CSV
# OUTPUT_FORMAT: (index, "AuthorId", "AuthorName", "Link")
##############################################################################################
def read_csv_list(name):
	with open(name) as f:
		data=[tuple(line) for line in csv.reader(f, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)]
		return data
		# lst = []
		# status.create_numbar(100, 10000000)
		# for line in csv.reader(f, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL):
		# 	status.update_numbar(len(lst), 10000000)
		# 	lst.append(tuple(line))
		# status.end_numbar()
		# return lst

##############################################################################################
# FUNCTION: make_query
# DESCRIPTION:  Makes a query to the database
# OUTPUT_FORMAT: (index, "AuthorId", "AuthorName", "Link")
##############################################################################################
def make_query(query):
	conn = psycopg2.connect(database="crimebb", user=db_username, password=db_password,  host="127.0.0.1", port="5432")
	#print("[DB] Extracting data")

	cur = conn.cursor()
	cur.execute(query)
	rows = cur.fetchall()
	conn.close()
	return rows

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
