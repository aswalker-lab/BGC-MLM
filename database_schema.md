```sql
CREATE TABLE TokenList (
    PFAM TEXT PRIMARY KEY,
    count INTEGER,
    prevalence INTEGER
);

CREATE TABLE GenomicMetadata (
    refseq_assembly TEXT PRIMARY KEY, 
    taxon TEXT, 
    source TEXT, 
    strain TEXT
);

CREATE TABLE RegionSequences (
    id INTEGER PRIMARY KEY AUTOINCREMENT, 
    refseq_assembly TEXT, 
    locus TEXT, 
    region INTEGER, 
    bgc_products TEXT, 
    tokenized_sequence TEXT
);
```
