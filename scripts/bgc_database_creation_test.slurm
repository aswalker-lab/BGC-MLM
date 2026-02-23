#!/bin/bash
#SBATCH --mail-user=joseph.cramer@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --account=walker_lab
#SBATCH --output=bgc_test_run.out

# Define file paths for the test run
INPUT_FILE="deduplicated_file_paths.txt"
OUTPUT_FILE="bgc_consolidated_database.db"
VENV_SOURCE=""

# Remove previous test file if it has the same name
if [[ -f "$OUTPUT_FILE" ]]; then
   rm $OUTPUT_FILE
fi

# Source the virtual environment, it may be necessary to add lines to load python before sourcing venv
source $VENV_SOURCE

# Optional: Print diagnostic information to the out file
echo "Starting test run on $(date)"
echo "Using input file: $INPUT_FILE"
echo "Outputting to: $OUTPUT_FILE"

# Execute the python script, passing the defined paths as arguments
python bgc_reprocessing.py --input "$INPUT_FILE" --output "$OUTPUT_FILE"

echo "Job completed on $(date)"
