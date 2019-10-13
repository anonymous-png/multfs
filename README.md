# A Methodology for Large-Scale Identification of Related Accounts in Underground Forums

This code presents the work developed for a large-scale analysis of Underground Forums. The scripts above allow the extraction of the data from a dataset and making use of that data, it is able to extract relationships between users.

## Getting started
This instructions would allow to execute the analysis on the dataset CrimeBB by Cambridge Cybercrime Centre.
### Prerequisites
The project is compiled and used under Python 3.7 version. It is not granted the functioning under other versions of the programming language. With the following command, all dependencies in requirements.txt should be installed:

```
conda create --name multfs --file requirements.txt
```

In case the database is PostgreSQL as in this case, the file ```db_credentials_example.py``` must be updated to include the credentials and renamed to ```db_credentials.py```

### MULTFS extraction
The execution of the methodology requires several steps before being executed. In this project we provide the code used for the specific use case. In the case of using different features, the function ```generate_args_dict```  in ```__init__.py``` should be modified to add the new features and its locations.

In the specific case of using the CrimeBB dataset, we perform the following steps. The first and fastest option is to execute:
```
python3 __init__.py alternative
```

However, this option does not include the computation of the thresholds that are necessary. For such purpose, we should execute the methodology step by step.

By executing the following command, we extract all the users associated to features:

```
python3 dataset_generators.py datasets
```

Before computing the pair similarities (coincidence and uniqueness), we need to evaluate the rank reduction by means of the optimal reduction graphs. Executing the following command, we generate the graph with the reductions in ```<feature>_files/<feature>_optimal_dist.pdf```:

```
python3 __init__.py optcsv
``` 

Finally, we can compute the pair similarities with:

```
python3 __init__.py cfs
```

Once it has been executed for all features, we can compute and execute the final snipet to obtain the MultFS in ```weighted_average.csv```:

```
python3 __init__.py multfs
```
___


