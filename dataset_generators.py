from common_utils import gen_csv_from_tuples, read_csv_list, make_query
import os, time, string, math
import itertools as it
import status, sys
def extract_user_to_link_csv(filename):
	tic = time.time()
	print("[+] Generating Link dataset")
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
  		regexp_matches( "Content", '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-\)+-Z^-_@.&+]|[!\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', 'g') AS "link"
  		FROM "Post" WHERE ("Site" != 4 OR "CitedPost"[1] = -1) AND "Content" ~ '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-\)+-Z^-_@.&+]|[!\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'),
		"B" AS (SELECT "Author", lower("link"[1]) as "link", count(*) as "repetitions" FROM "A" GROUP BY "Author", "link" )
		SELECT "B"."Author",
		string_agg("B"."link", ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""	
	rows = make_query(query)
	rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != str(-1)]
	gen_csv_from_tuples(filename, ["IdAuthor", "link"], rows)
	print("[+] Finished generating Link dataset in %d" % (time.time() - tic))


def extract_user_to_ip_csv(filename):
	tic = time.time()
	print("[+] Generating IP Address dataset")
	query= """WITH "A" AS (SELECT
		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
		regexp_matches( "Content", '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', 'g') AS "ip"
		FROM "Post"	WHERE ("Site" != 4 OR "CitedPost"[1] = -1) AND "Content" ~ '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'),
		"B" AS (SELECT "Author", "ip", count(*) as "repetitions" FROM "A" GROUP BY "Author", "ip" )
		SELECT "B"."Author",
		string_agg("B"."ip"[1] || '.' ||"B"."ip"[2] || '.' ||"B"."ip"[3]|| '.' ||"B"."ip"[4], ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""
	rows = make_query(query)
	rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != str(-1)]
	gen_csv_from_tuples(filename, ["IdAuthor", "IP"], rows)
	print("[+] Finished generating IP Address dataset in %d" % (time.time() - tic))

def extract_user_to_email_csv(filename):
	tic = time.time()
	print("[+] Generating Email dataset")
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
  		regexp_matches( "Content", '(?:(?![*]))([A-Za-z0-9\._%-\)\+]+@[A-Za-z0-9\.-]+[.][A-Za-z]+)', 'g') AS "email"
  		FROM "Post" WHERE ("Site" != 4 OR "CitedPost"[1] = -1) AND "Content" ~ '(?:(?![*]))([A-Za-z0-9\._%-\)\+]+@[A-Za-z0-9\.-]+[.][A-Za-z]+)'),
		"B" AS (SELECT "Author", lower("email"[1]) as "email", count(*) as "repetitions" FROM "A" GROUP BY "Author", "email" )
		SELECT "B"."Author",
		string_agg("B"."email", ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""
	rows = make_query(query)
	rows = [list(row[:1] + tuple([x for x in row[1].split(", ")],)) for row in rows if row[0] != str(-1)]
	for row in range(len(rows)):
		for col in range(1, len(rows[row])):
			if len(rows[row][col]) > len("***LINK***") and rows[row][col][:len("***LINK***")] == "***LINK***":
				rows[row][col] = rows[row][col][len("***LINK***"):]	

	for row in range(len(rows)):
		rows[row] = (rows[row][0],) + tuple(set(rows[row][1:]))
	gen_csv_from_tuples(filename, ["IdAuthor", "email"], rows)
	print("[+] Finished generating Email dataset in %d" % (time.time() - tic))

def extract_user_to_skype_csv(filename):
	tic = time.time()
	print("[+] Generating Skype dataset")
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author", 
  		regexp_matches( "Content", 'skype\s*:\s*([a-zA-Z0-9:\.]{1,37})', 'g') AS "skype"
  		FROM "Post" WHERE ("Site" != 4 OR "CitedPost"[1] = -1) AND "Content" ~ 'skype\s*:\s*([a-zA-Z0-9:\.]{1,37})'),
		"B" AS (SELECT "Author", lower("skype"[1]) as "skype", count(*) as "repetitions" FROM "A" GROUP BY "Author", "skype" )
		SELECT "B"."Author",
		string_agg("B"."skype", ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""
	rows = make_query(query)
	rows = [list(row[:1] + tuple([x for x in row[1].split(", ")],)) for row in rows if row[0] != str(-1)]
	for row in range(len(rows)):
		for col in range(1, len(rows[row])):
			if rows[row][col][-1] == '.':
				rows[row][col] = rows[row][col][:-1]
				
	for row in range(len(rows)):
		rows[row] = (rows[row][0],) + tuple(set(rows[row][1:]))

	gen_csv_from_tuples(filename, ["IdAuthor", "skype"], rows)
	print("[+] Finished generating Skype dataset in %d" % (time.time() - tic))

def extract_user_to_btc_csv(filename):
	tic = time.time()
	print("[+] Generating Bitcoin dataset")
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
  		regexp_matches( "Content", '\y([13][a-km-zA-HJ-NP-Z1-9]{25,34})\y', 'g') AS "btc"
  		FROM "Post" WHERE ("Site" != 4 OR "CitedPost"[1] = -1) AND "Content" ~ '([13][a-km-zA-HJ-NP-Z1-9]{25,34})'),
		"B" AS (SELECT "Author", lower("btc"[1]) as "btc", count(*) as "repetitions" FROM "A" GROUP BY "Author", "btc" )
		SELECT "B"."Author",
		string_agg("B"."btc", ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""

	rows = make_query(query)
	rows = [list(row[:1] + tuple([x for x in row[1].split(", ")],)) for row in rows if row[0] != str(-1)]
	for row in range(len(rows)):
		for col in range(1, len(rows[row])):
			if rows[row][col][-1] == '.':
				rows[row][col] = rows[row][col][:-1]
				
	for row in range(len(rows)):
		rows[row] = (rows[row][0],) + tuple(set(rows[row][1:]))
	gen_csv_from_tuples(filename, ["IdAuthor", "btc"], rows)
	print("[+] Finished generating Bitcoin dataset in %d" % (time.time() - tic))

def create_dir(filename):
	try:
	    # Create target Directory
	    os.mkdir(filename)
	    #print("Directory " , filename ,  " Created ") 
	except FileExistsError:
	    print("Directory " , filename ,  " already exists")
	    
def create_directories_and_datasets():
	create_dir('btc_files')
	extract_user_to_btc_csv("btc_files/user_to_btc.csv")
	create_dir('email_files')
	extract_user_to_email_csv("email_files/user_to_email.csv")
	create_dir('ip_files')
	extract_user_to_ip_csv("ip_files/user_to_ip.csv")
	create_dir('skype_files')
	extract_user_to_skype_csv("skype_files/user_to_skype.csv")
	create_dir('link_files')
	extract_user_to_link_csv("link_files/user_to_link.csv")

def create_directories_and_datasets_1():
	create_dir('btc_files')
	extract_user_to_btc_csv("btc_files/user_to_btc.csv")
	create_dir('email_files')
	extract_user_to_email_csv("email_files/user_to_email.csv")
	create_dir('ip_files')
	extract_user_to_ip_csv("ip_files/user_to_ip.csv")
	create_dir('skype_files')
	extract_user_to_skype_csv("skype_files/user_to_skype.csv")

def create_directories_and_datasets_2():
	create_dir('link_files')
	extract_user_to_link_csv("link_files/user_to_link.csv")
	
def get_user_site(user):
	pos_bracket = user.find("[")
	user_id = int(user[:pos_bracket])
	site_id = int(user[pos_bracket + 1:-1])
	return user_id, site_id

def generate_user_dataset(user, uind, total):
	print("[-] Going for user %d/%d - %s" %(uind, total,  user))
	tic = time.time()
	user_id, site_id = get_user_site(user)
	print("[- -] Extracting from DB")
	query = """SELECT "IdPost", "Content" FROM "Post" WHERE "Author" = %d AND "Site" = %d;""" % (user_id, site_id)
	rows = make_query(query)
	rows = [(row[0], row[1]) for row in rows]
	print("[+ +] Done extracting from DB")
	#a = string.ascii_lowercase
	#b = math.ceil(float(len(rows)) / float(len(a)) )
	#names = ["".join(elem) for iter in [it.product(a, repeat=i) for  i in range(1,b + 1)] for elem in iter]
	directory = 'Author/' + user + "/"
	create_dir(directory)
	print("[- -] Generating files for user, total: %d" % (len(rows)))
	for ind, content in enumerate(rows):
		filename = str(user_id) + "-" + str(site_id) + "-" + str(content[0])
		with open(directory + filename + ".txt", 'w+') as file:  
			file.write(content[1])
	print("[+ +] Generating files for user, total: %d" % (len(rows)))
	print("[+] Going for user %d - %s" %(uind, user))

def generate_directories_for_users():
	print("[>] Creating dir")
	create_dir("Author/")
	print("[>] Reading user csv list")
	lst_users = read_csv_list("weighted_average.csv")[1:]

	#lst_users = [(x[0], x[1] for x in lst_users if float(x[2]) < 0.35)
	ev_set = set()
	for entry in lst_users:
		if float(entry[2]) >= 0.35:
			break
		ev_set.add(entry[0])
		ev_set.add(entry[1])
	#status.create_numbar(100, len(ev_set))
	for ind, user in enumerate(ev_set):
		#status.update_numbar(ind, len(ev_set))
		generate_user_dataset(user, ind, len(ev_set))
	#status.end_numbar()

def main():
	if len(sys.argv) < 2:
		print("""Usage: python3 dataset_generators.py <option>
	<option>:
		'datasets': generate datasets and directories
		'authorfolders': generate folders for users
		""")
		return

	if sys.argv[1] == 'authorfolders':
		print("[>>] Doing all")
		generate_directories_for_users()

	elif sys.argv[1] == 'datasets':
		print("[>>] NONE")
		create_directories_and_datasets()
if __name__ == "__main__":
	main()


