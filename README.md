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
