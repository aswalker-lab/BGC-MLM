import os
import re
import sqlite3
import time
from datetime import timedelta
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple
from Bio import SeqIO
import argparse
from tqdm import tqdm
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description="Process BGC GenBank files.")
parser.add_argument("-i", "--input", default="gbk_file_paths.txt", help="Input text file containing paths")
parser.add_argument("-o", "--output", default="bgc_embeddings.db", help="Output SQLite database name")
args = parser.parse_args()

# --- CONFIGURATION ---
FILE_PATHS_TXT = args.input
DATABASE_NAME = args.output
SHORT_GAP_THRESHOLD = 50   # Base pairs
LONG_GAP_THRESHOLD = 500   # Base pairs
OVERLAP_FRACTION = 0.5     # 50% overlap

# --- PASS 1: FAST PREVALENCE COUNTING ---
def pass1_worker(path: str) -> Dict[str, int]:
    """ Reads a single file and returns a dict of PFAM counts. """
    if "region" not in os.path.basename(path):
        return None
        
    pfam_regex = re.compile(r'(PF\d{5})')
    counts = {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                for pfam in pfam_regex.findall(line):
                    counts[pfam] = counts.get(pfam, 0) + 1
        return counts
    except Exception:
        return None

def calculate_global_prevalence(file_paths: List[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Rapidly scans only region files using regex to count global PFAM prevalence and total counts.
    Bypassing Biopython here saves hours/days of compute time.
    Uses multiprocessing.
    """
    print("Starting Pass 1: Calculating global PFAM prevalence and counts...")
    prevalence = {}
    counts = {}
    
    with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        # Added tqdm for Pass 1. mininterval=15 ensures Slurm logs are updated roughly 4 times a minute
        for result in tqdm(pool.imap_unordered(pass1_worker, file_paths, chunksize=100), 
                           total=len(file_paths), 
                           desc="Pass 1 (Regex)", 
                           mininterval=15.0, 
                           ascii=True, 
                           file=sys.stdout):
            if result:
                for pfam, count in result.items():
                    counts[pfam] = counts.get(pfam, 0) + count
                    prevalence[pfam] = prevalence.get(pfam, 0) + 1
            
    print(f"Pass 1 Complete. Found {len(prevalence)} unique PFAMs.")
    return prevalence, counts

# --- PASS 2: WORKER LOGIC ---
def resolve_overlap(feat1: Dict, feat2: Dict, prevalence: Dict[str, int]) -> Dict:
    """
    Resolves overlaps between two domains based on user-defined prevalence rules.
    Returns the domain that should be kept.
    """
    pfam1, pfam2 = feat1['pfam'], feat2['pfam']
    prev1, prev2 = prevalence.get(pfam1, 0), prevalence.get(pfam2, 0)
    
    # Rule 1: < 1000 excluded if other is much higher (assumed >= 1000)
    if prev1 < 1000 and prev2 >= 1000: return feat2
    if prev2 < 1000 and prev1 >= 1000: return feat1
    
    # Rule 2: Encompassing
    len1 = feat1['end'] - feat1['start']
    len2 = feat2['end'] - feat2['start']
    
    f1_encompasses = (feat1['start'] <= feat2['start'] and feat1['end'] >= feat2['end'])
    f2_encompasses = (feat2['start'] <= feat1['start'] and feat2['end'] >= feat1['end'])
    
    if f1_encompasses:
        return feat2 if prev2 >= 2000 else feat1
    if f2_encompasses:
        return feat1 if prev1 >= 2000 else feat2
        
    # Fallback: keep the larger domain, or higher prevalence if same size
    if len1 != len2:
        return feat1 if len1 > len2 else feat2
    return feat1 if prev1 >= prev2 else feat2

def process_directory(directory: str, files: List[str], prevalence: Dict[str, int]) -> Tuple[Dict, List[Dict]]:
    """ Worker function to process all files in a single terminal directory. """
    genomic_file = next((f for f in files if "genomic" in os.path.basename(f) and "region" not in os.path.basename(f)), None)
    region_files = [f for f in files if "region" in os.path.basename(f)]
    
    genomic_data = {"refseq": "UNKNOWN", "taxon": "UNKNOWN", "source": "UNKNOWN", "strain": "UNKNOWN"}
    
    # 1. Parse Genomic File for Metadata
    if genomic_file:
        try:
            record = next(SeqIO.parse(genomic_file, "genbank"))
            # Attempt to extract RefSeq from id or db_xref
            for xref in record.dbxrefs:
                if "Assembly:" in xref:
                    genomic_data['refseq'] = xref[9:]
            
            if 'taxonomy' in record.annotations:
                genomic_data['taxon'] = "; ".join(record.annotations['taxonomy'])

            if 'source' in record.annotations:
                genomic_data['source'] = record.annotations['source']
            
            # Extract strain from source feature
            for feat in record.features:
                if feat.type == "source":
                    genomic_data['strain'] = feat.qualifiers.get('strain', ['UNKNOWN'])[0]
                    break
        except Exception as e:
            # Replaced print with pass or logging to prevent log file spam during multiprocessing
            pass # Fail gracefully, fall back to UNKNOWN

    region_data_list = []
    
    # 2. Parse Region Files
    for r_file in region_files:
        try:
            record = SeqIO.read(r_file, "genbank")
            locus = record.name
            raw_domains = []
            bgc_products = []
            
            # Extract domains
            for feat in record.features:
                pfam_id = None
                
                if feat.type == "cand_cluster" and bgc_products == []:
                    bgc_products = feat.qualifiers.get('product', ['ERR'])
                    continue

                # Search for PFAM ID in qualifiers using regex for robustness
                for db_xref in feat.qualifiers.get('db_xref', []):
                    match = re.search(r'(PF\d{5})', db_xref)
                    if match:
                        pfam_id = match.group(1)
                        break
                        
                if pfam_id:
                    locus_tag = feat.qualifiers.get('locus_tag', [None])[0]
                    raw_domains.append({
                        'pfam': pfam_id,
                        'start': int(feat.location.start),
                        'end': int(feat.location.end),
                        'gene': locus_tag
                    })
            
            if not raw_domains:
                continue
                
            # Sort strictly by genomic coordinate
            raw_domains.sort(key=lambda x: x['start'])
            
            # 3. Resolve Overlaps
            resolved_domains = []
            for current in raw_domains:
                if not resolved_domains:
                    resolved_domains.append(current)
                    continue
                    
                prev = resolved_domains[-1]
                overlap = max(0, min(prev['end'], current['end']) - max(prev['start'], current['start']))
                
                if overlap > 0:
                    shorter_len = min(prev['end'] - prev['start'], current['end'] - current['start'])
                    if shorter_len > 0 and (overlap / shorter_len) > OVERLAP_FRACTION:
                        # Resolve and replace the last accepted domain with the winner
                        winner = resolve_overlap(prev, current, prevalence)
                        resolved_domains[-1] = winner
                        continue
                
                resolved_domains.append(current)
                
            # 4. Tokenize with Gaps and Genes
            tokens = []
            last_gene = None
            last_end = 0
            
            for d in resolved_domains:
                if last_gene is not None:
                    if d['gene'] != last_gene:
                        tokens.append("<gene>")
                    else:
                        gap = d['start'] - last_end
                        if gap > LONG_GAP_THRESHOLD:
                            tokens.append("<long-gap>")
                        elif gap > SHORT_GAP_THRESHOLD:
                            tokens.append("<short-gap>")
                            
                tokens.append(d['pfam'])
                last_gene = d['gene']
                last_end = d['end']
            
            region_id = 0
            match = re.search(r'region(\d{3})', os.path.basename(r_file))
            if match:
                region_id = match.group(1)

            region_data_list.append({
                "locus": locus,
                "refseq": genomic_data['refseq'],
                "tokenized": ", ".join(tokens),
                "region": int(region_id),
                "bgc_products": ", ".join(bgc_products)
            })
            
        except Exception as e:
            continue
            
    return genomic_data, region_data_list

def worker_process(directory: str, files: List[str], prevalence: Dict[str, int], queue: mp.Queue):
    """ Worker entry point: processes a directory and puts results in the DB queue. """
    genomic_data, region_data_list = process_directory(directory, files, prevalence)
    queue.put(("GENOMIC", genomic_data))
    for reg in region_data_list:
        queue.put(("REGION", reg))

def worker_process_wrapper(args_tuple):
    """ 
    Wrapper function required to unpack arguments for pool.imap_unordered 
    which is necessary for dynamic tqdm updates.
    """
    return worker_process(*args_tuple)

# --- DATABASE WRITER THREAD ---
def database_writer(queue: mp.Queue):
    """ Runs in a separate process, strictly pulling from the queue to avoid SQLite locks. """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS GenomicMetadata
                     (refseq_assembly TEXT PRIMARY KEY, taxon TEXT, source TEXT, strain TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS RegionSequences
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, refseq_assembly TEXT, locus TEXT, region INTEGER, bgc_products TEXT, tokenized_sequence TEXT)''')
    
    conn.commit()
    batch_size = 1000
    genomic_batch = []
    region_batch = []
    
    while True:
        msg = queue.get()
        if msg == "DONE":
            break
            
        m_type, data = msg
        if m_type == "GENOMIC":
            genomic_batch.append((data['refseq'], data['taxon'], data['source'], data['strain']))
        elif m_type == "REGION":
            region_batch.append((data['refseq'], data['locus'], data['region'], data['bgc_products'], data['tokenized']))
            
        if len(region_batch) >= batch_size:
            cursor.executemany("INSERT OR IGNORE INTO GenomicMetadata (refseq_assembly, taxon, source, strain) VALUES (?, ?, ?, ?)", genomic_batch)
            cursor.executemany("INSERT INTO RegionSequences (refseq_assembly, locus, region, bgc_products, tokenized_sequence) VALUES (?, ?, ?, ?, ?)", region_batch)
            conn.commit()
            genomic_batch.clear()
            region_batch.clear()
            
    # Final commit - Separated so genomic data commits even if regions are empty
    if genomic_batch:
        cursor.executemany("INSERT OR IGNORE INTO GenomicMetadata (refseq_assembly, taxon, source, strain) VALUES (?, ?, ?, ?)", genomic_batch)
    if region_batch:
        cursor.executemany("INSERT INTO RegionSequences (refseq_assembly, locus, region, bgc_products, tokenized_sequence) VALUES (?, ?, ?, ?, ?)", region_batch)
    conn.commit()
    conn.close()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    starttime = time.perf_counter()
    with open(FILE_PATHS_TXT, 'r') as f:
        all_files = [line.strip() for line in f if line.strip()]
        
    # Group files by their parent directory
    dir_to_files = {}
    for filepath in all_files:
        parent = os.path.dirname(filepath)
        if parent not in dir_to_files:
            dir_to_files[parent] = []
        dir_to_files[parent].append(filepath)

    # Pass 1: Global Prevalence and Counts
    prevalence_dict, counts_dict = calculate_global_prevalence(all_files)
    
    # Write TokenList to database
    print("Writing TokenList to database...")
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS TokenList
                     (PFAM TEXT PRIMARY KEY, count INTEGER, prevalence INTEGER)''')
    token_data = [(pfam, counts_dict[pfam], prevalence_dict[pfam]) for pfam in prevalence_dict]
    cursor.executemany("INSERT OR REPLACE INTO TokenList (PFAM, count, prevalence) VALUES (?, ?, ?)", token_data)
    conn.commit()
    conn.close()
    
    # Pass 2: Multiprocessing Setup
    manager = mp.Manager()
    db_queue = manager.Queue()
    
    # Start the DB writer thread
    writer_process = mp.Process(target=database_writer, args=(db_queue,))
    writer_process.start()
    
    # Dispatch workers
    print(f"\nStarting Pass 2: Processing {len(dir_to_files)} directories across {mp.cpu_count()} cores...")
    
    # Package arguments for the wrapper
    worker_args = [(directory, files, prevalence_dict, db_queue) for directory, files in dir_to_files.items()]
    
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        # Use imap_unordered to allow tqdm to update as tasks finish.
        # mininterval=15.0 and ascii=True prevent massive, distorted Slurm log files.
        for _ in tqdm(pool.imap_unordered(worker_process_wrapper, worker_args), 
                      total=len(worker_args), 
                      desc="Pass 2 (Biopython)", 
                      mininterval=15.0, 
                      ascii=True, 
                      file=sys.stdout):
            pass
        
    # Shutdown DB writer
    db_queue.put("DONE")
    writer_process.join()
    print(f"\nPipeline Complete. Data saved to {DATABASE_NAME} ----- Time taken {timedelta(seconds=time.perf_counter()-starttime)}")