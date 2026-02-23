#!/bin/bash
#SBATCH --mail-user=**insert email here**
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm_output/bgc_database_creation_%j.out

# Define file paths for the run
INPUT_FILE="deduplicated_file_paths.txt"
OUTPUT_FILE="bgc_consolidated_database.db"
VENV_SOURCE=""
# Source the virtual environment, it may be necessary to add lines to load python before sourcing venv
source $VENV_SOURCE

# Optional: Print diagnostic information to the out file
echo "Starting test run on $(date)"
echo "Using input file: $INPUT_FILE"
echo "Outputting to: $OUTPUT_FILE"

# Execute the python script, passing the defined paths as arguments
python bgc_reprocessing.py --input "$INPUT_FILE" --output "$OUTPUT_FILE"

echo "Job completed on $(date)"
