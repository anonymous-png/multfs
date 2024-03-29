import re
from common_utils import gen_csv_from_tuples
import numpy as np
import math
import status
import time

def ip_value_removal(dictio_of_users, dictio_of_values):
	print("[+] Removing reserved IP addresses...")
	value_list = []
	for ip in dictio_of_values.keys():
		if re.search(r'192\.168\.\d{1,3}\.\d{1,3}', 
			ip,  re.IGNORECASE | re.DOTALL | re.VERBOSE | re.MULTILINE):
			value_list.append(ip)
		elif re.search(r'172\.\d{1,3}\.\d{1,3}\.\d{1,3}', 
			ip,  re.IGNORECASE | re.DOTALL | re.VERBOSE | re.MULTILINE):
			value_list.append(ip)
		elif re.search(r'10\.\d{1,3}\.\d{1,3}\.\d{1,3}', 
			ip,  re.IGNORECASE | re.DOTALL | re.VERBOSE | re.MULTILINE):
			value_list.append(ip)
		elif ip == '127.0.0.1':
			value_list.append(ip)
	return value_list

def ip_user_removal(dictio_of_users, dictio_of_values):
	print("[-] Removing IP addresses appearing more than 20 times")
	return [key for key, values in dictio_of_users.items() if len(values) > 20]

def skype_user_removal(dictio_of_users, dictio_of_values):
	print("[-] Removing Skype appearing more than 5 times")
	return [key for key, values in dictio_of_users.items() if len(values) > 5]

def email_user_removal(dictio_of_users, dictio_of_values):
	print("[-] Removing emails appearing more than 5 times")
	return [key for key, values in dictio_of_users.items() if len(values) > 5]

def link_value_removal_2(dictio_of_users, dictio_of_values):
	print("[+] Highlighting links to other forums...")
	dictio_of_sites = {
		"hackforums.net": 0,
		"mpgh.net": 4,
		"raidforums.com": 12,
		"antichat.ru": 10,
		"blackhatworld.com": 8,
		"garage4hackers.com": 7,
		"greysec.net": 6,
		"stresserforums.net": 5,
		"kernelmode.info": 1,
		"safeskyhacks.com": 13,
		"offensivecommunity.net": 3
	}
	link_list = []
	count_external_refs = 0
	status.create_numbar(100, len(dictio_of_values))
	for index, elem in enumerate(dictio_of_values.items()):
		status.update_numbar(index, len(dictio_of_values))
		link, users = elem[0], elem[1]
		for link_site, site_num in dictio_of_sites.items():
			if link_site in link:
				#print(link_site, link)
				ext_reference = False
				for user in users:
					user_site = int(user[len(user) - user[::-1].find('['):-1])
					if site_num != user_site:
						#print(users)
						ext_reference = True
						break
				if not ext_reference:
					link_list.append(link)
				break

	status.end_numbar()
	return link_list
def link_value_removal_keep_params(dictio_of_users, dictio_of_values):
	print("Removing short links...")
	link_list = []
	for link in dictio_of_values.keys():
		if link[:-1].count("/") <= 2 and not ".onion" in  link:
			link_list.append(link)
	return link_list
def link_value_removal(dictio_of_users, dictio_of_values):
	print("[+] Removing local links...")
	dictio_of_sites = {
		"raidforums.com": 12,
		"antichat.ru": 10,
		"hackforums.net": 0,
		"blackhatworld.com": 8,
		"mpgh.net": 4,
		"garage4hackers.com": 7,
		"greysec.net": 6,
		"stresserforums.net": 5,
		"kernelmode.info": 1,
		"safeskyhacks.com": 13,
		"offensivecommunity.net": 3
	}
	local_links = []
	for link in dictio_of_values.keys():
		is_link_internal = False
		for site in dictio_of_sites.keys():
			if site in link:
				is_link_internal = True
				break
		if is_link_internal == True:
			local_links.append(link)
	return local_links

def link_user_removal(dictio_of_users, dictio_of_values):
	print("[-] Removing users with activity below average")
	lst = [len(v) for k, v in dictio_of_users.items()]
	v = np.array(lst)
	mean = math.ceil(np.mean(v))
	std = math.ceil(np.std(v))
	return [user for user, values in dictio_of_users.items() if len(values) <= mean + (0.25 * std) ]
