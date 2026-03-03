# BGC-MLM
masked language foundation model for biosynthetic gene clusters

# Functionalities
- Centralized parsing for CLI arguments
- Generic dataset creation for classification, regression, and metric learning tasks
- Flexible generic code for architecture, training, and evaluation

# Setup and Usage
### Environment Setup
Using the pip-tools package (which can be installed with pip install pip-tools) in a virtual environment run:
 ```console
 pip-compile --output-file=requirements.txt requirements.in
 pip install -r requirements.txt
 ```

### Training
Use scripts from the `scripts/` directory to train the model with settings and inputs desired.

### Testing
Run tests using pytest:
 ```console
 python -m pytest tests/ -v --tb=short
 ```

### Dataset Creation
To create SQLite3 database run
'''console
python bgc_reprocessing.py --input *INPUT_FILE* --output *OUTPUT_FILE*
'''
For convenience there is a slurm script provided --> scripts/bgc_database_creation.slurm
Change the paths for INPUT_FILE, OUTPUT_FILE, and VENV_SOURCE to the respective paths so that the script can access the correct data and modify the header with email, log file, and resources as desired.
For the input file use a text file where each line is the path to a gbk file. These files must have the terms "genomic" for full genome files containing metadata for species, taxon, source, and strain; or "region" for files with the region specific PFAM domains.
For the output file give the name and path where you want the new database to be created.
For the Venv give the path to the virtual environment created for the project.
