#!/usr/bin/env python3
import pandas as pd

# Filenames
ENZ_XLSX     = '41467_2025_61530_MOESM4_ESM.xlsx'
SUB_XLSX     = '41467_2025_61530_MOESM5_ESM.xlsx'
SCR_XLSX     = '41467_2025_61530_MOESM6_ESM.xlsx'
OUT_TSV_FULL = 'merged_full.tsv'
OUT_TSV_POS  = 'merged_pos_only.tsv'

# 1. Load enzyme metadata
enz_df = pd.read_excel(
    ENZ_XLSX, sheet_name=0, dtype=str
)[[
    'Enzyme group','Enzyme Name','Gene AGI','Uniprot Entry','Protein sequence'
]].rename(columns={
    'Enzyme group':'enzyme_group',
    'Enzyme Name':'enzyme',
    'Gene AGI':'gene_agi',
    'Uniprot Entry':'uniprot_entry',
    'Protein sequence':'sequence'
})
# Remove the control enzyme
enz_df = enz_df[enz_df['enzyme'] != 'ALG14LP']

# 2. Load substrate metadata (unique acceptors)
sub_df = pd.read_excel(
    SUB_XLSX, sheet_name='Substrates_VB_clean', dtype=str
)[[
    'Substrate name','InchiKey','InchiKey_MoNA','SMILES','CSMILES','superclass'
]].rename(columns={
    'Substrate name':'acceptor',
    'InchiKey':'inchikey',
    'InchiKey_MoNA':'inchikey_mona',
    'SMILES':'smiles',
    'CSMILES':'csmiles'
})
sub_df = sub_df.drop_duplicates(subset='acceptor')

# 3. Build full enzyme×substrate grid
enz_list = enz_df['enzyme'].drop_duplicates()
sub_list = sub_df['acceptor'].drop_duplicates()
full = pd.MultiIndex.from_product(
    [enz_list, sub_list],
    names=['enzyme','acceptor']
).to_frame(index=False)

# 4. Merge static metadata onto full grid
full = full.merge(enz_df, on='enzyme', how='left')
full = full.merge(sub_df, on='acceptor', how='left')

# 5. Load and preprocess positive hits (0.85 threshold)
hits = pd.read_excel(
    SCR_XLSX,
    sheet_name='Screen_CosineScore_0.85',
    dtype={
        'Enzyme_name': str,
        'Name':         str,
        'CSMILES':      str,
        'CosineScore_single': str,
        'CosineScore_double': str
    }
)[[
    'Enzyme_name',
    'Name',
    'CSMILES',
    'CosineScore_single',
    'CosineScore_double'
]].rename(columns={
    'Enzyme_name':        'enzyme',
    'Name':               'acceptor',
    'CSMILES':            'csmiles',
    'CosineScore_single': 'cs_single',
    'CosineScore_double': 'cs_double'
})

# 5a. Prepend 'UGT' prefix to enzyme IDs
hits['enzyme'] = hits['enzyme'].str.strip().apply(lambda x: f'UGT{x}')

# 5b. Strip whitespace from acceptor and csmiles
hits['acceptor'] = hits['acceptor'].str.strip()
hits['csmiles']  = hits['csmiles'].str.strip()

# 6. Deduplicate hits at (enzyme, acceptor) level
hits_unique = hits.drop_duplicates(subset=['enzyme', 'acceptor'])

# 7. Merge hit scores onto full grid  ← drop csmiles from the join keys
merged_full = full.merge(
    hits_unique[['enzyme', 'acceptor', 'cs_single', 'cs_double']],
    on=['enzyme', 'acceptor'],
    how='left'
)

# 8. Write out
merged_full.to_csv(OUT_TSV_FULL, sep='\t', index=False)
print(f"Wrote full grid to {OUT_TSV_FULL} ({len(merged_full)} rows)")

