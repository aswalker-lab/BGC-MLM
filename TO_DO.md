# TO DO

- write up UML Diagram as a guide for edits and reformatting of codebase
- refactor codebase
    - central Parser
    - split BGC_MLM_tools into multiple files
    - make dataset creation a standalone file and generic
    - seperate codebase into folders for the different tasks
    - variable and file naming conventions
- write new read me documentation
- write shell scripts to run code
- provide environment dependencies as yaml or requirements.txt
- setup tests to run shell scripts and validate pipeline still works, to verify refactoring and code edits don't break functionality

Notes
- setup acchre for data access
- full protein embedding worry about cheating on domains
- biopython to read in genbank
- files only have one line so max bgc length 
- bioactivity from classifiers (bioactivity)
- frozen, pretrain, and from scratch
- property prediction
- functional group prediction
- for downstream prediction pfams -> truth labels
- BGC and product linking using similarity of fingerprints for the error
- auxiliary losses with product description before prediction