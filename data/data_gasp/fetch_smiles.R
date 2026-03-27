#!/usr/bin/env Rscript

# fetch_smiles.R
# Reads PubChem CIDs from stdin, fetches canonical SMILES via webchem,
# and writes a two-column TSV (cid\tsmiles) to stdout.

# Install/load dependencies
if (!requireNamespace("webchem", quietly = TRUE)) {
  install.packages("webchem", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table", repos = "https://cloud.r-project.org")
}

library(webchem)
library(data.table)

# Read CIDs from stdin (one per line)
cids <- readLines(file("stdin"))

# Fetch SMILES properties from PubChem
res <- pc_prop(cids, properties = "CanonicalSMILES")

# Convert to data.table and normalize types
dt_res <- as.data.table(res)
# Rename columns: first->cid, second->smiles
dt_res_cols <- names(dt_res)
if (length(dt_res_cols) >= 2) {
  setnames(dt_res, old = dt_res_cols[1], new = "cid", skip_absent = TRUE)
  setnames(dt_res, old = dt_res_cols[2], new = "smiles", skip_absent = TRUE)
}
# Ensure cid is character to match input cids
dt_res[, cid := as.character(cid)]

# Prepare base table of all input CIDs in correct order
dt_in <- data.table(cid = as.character(cids))

# Merge to preserve all input CIDs and order
dt_out <- merge(dt_in, dt_res[, .(cid, smiles)], by = "cid", all.x = TRUE)

# Write TSV to stdout
fwrite(dt_out, sep = "\t", quote = FALSE, col.names = TRUE)

