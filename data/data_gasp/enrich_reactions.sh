#!/usr/bin/env bash
set -euo pipefail

# Paths (adjust if needed)
GASP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../GASP && pwd)"
REACT="$GASP_ROOT/data/reactions/reactions.tsv"
SEQS="$GASP_ROOT/results/06-unaligned/sequences.tsv"

WORKDIR="$(pwd)"  # assume running in ESP/data/GASP_data
SMILES_TSV="$WORKDIR/smiles.tsv"
OUT_TSV="$WORKDIR/reactions_with_smiles_seq.tsv"

# 1. Extract CIDs
mlr --tsv cut -f cid "$REACT" \
  | tail -n +2 \
  | sort -u \
  > cids.txt

# 2. Fetch SMILES
cat cids.txt | ./fetch_smiles.R > "$SMILES_TSV"

# 3. Join SMILES into reactions
mlr --tsv \
  join -j cid -f "$SMILES_TSV" \
  then unsparsify \
  "$REACT" \
  > step1.tsv

# 4. Join enzyme sequences
mlr --tsv \
  join -j enzyme -f "$SEQS" \
  then unsparsify \
  step1.tsv \
  > "$OUT_TSV"

# 5. Cleanup
rm step1.tsv cids.txt

echo "✔ Enriched table at $OUT_TSV"

