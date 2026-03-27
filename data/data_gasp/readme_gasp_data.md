# GASP Data Enrichment Pipeline

This folder contains scripts and outputs for enriching the GASP `reactions.tsv` dataset with PubChem SMILES strings and enzyme sequences.

## Contents

- **fetch\_smiles.R**: Rscript to fetch canonical SMILES from PubChem for a list of CIDs.
- **enrich\_reactions.sh**: Bash driver that:
  1. Extracts unique CIDs from `reactions.tsv`
  2. Uses `fetch_smiles.R` to obtain SMILES
  3. Joins SMILES and enzyme sequences into the final TSV
- **test\_cids.txt**: Sample list of three CIDs (2244, 5202, 5793) for testing.
- **test\_smiles.tsv**: SMILES results for `test_cids.txt`.
- **smiles.tsv**: SMILES results for all CIDs in the GASP `reactions.tsv`.
- **reactions\_with\_smiles\_seq.tsv**: Final enriched table with original reaction data plus `smiles` and `sequence` columns.

## Prerequisites

- **System Tools**:

  - Bash shell
  - Miller (`mlr`)
  - R (with `webchem` and `data.table` packages)

- **Directory Structure**: This folder should be located under `ESP/data/GASP_data`. The GASP repository root should be at `ESP/`, so that `../GASP/data/reactions/reactions.tsv` and `../GASP/results/06-unaligned/sequences.tsv` exist.

## Usage

1. **Make scripts executable**:

   ```bash
   chmod +x fetch_smiles.R enrich_reactions.sh
   ```

2. **Run the enrichment**:

   ```bash
   cd ~/thesis/repos/ESP/data/GASP_data
   ./enrich_reactions.sh
   ```

   - This produces `smiles.tsv` and `reactions_with_smiles_seq.tsv`.

3. **Verify output**:

   ```bash
   head reactions_with_smiles_seq.tsv
   ```

4. **Test the SMILES fetcher** (optional):

   ```bash
   echo -e "2244\n5202\n5793" > test_cids.txt
   ./fetch_smiles.R < test_cids.txt > test_smiles.tsv
   cat test_smiles.tsv
   ```

## File Descriptions

- **fetch\_smiles.R**: Reads CIDs from stdin, queries PubChem via `webchem::pc_prop()`, and outputs a TSV of `cid\tsmiles`.
- **enrich\_reactions.sh**: Orchestrates extraction of CIDs, SMILES fetching, and joining with enzyme sequences.
- **smiles.tsv**: SMILES for all CIDs in the GASP reactions dataset.
- **reactions\_with\_smiles\_seq.tsv**: Final enriched dataset, containing all original reaction fields plus `smiles` (PubChem SMILES) and `sequence` (enzyme sequence).
- **test\_cids.txt** / **test\_smiles.tsv**: Example files for quick end-to-end testing.

## Notes & Next Steps

- **Missing Values**: All CIDs in `smiles.tsv` have valid SMILES; no missing entries.
- **Reproducibility**: Record `webchem` and `data.table` versions via `sessionInfo()` in `fetch_smiles.R` if needed.
- **Performance**: For large CID sets, consider caching results or parallelizing queries.

---

*Generated on \$(date)*

