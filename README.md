# RETECO

> Temporal Grounded Retrieval

## Sub-track 1a — Temporal Retrieval
Input
Query + corpus
A complex temporal information need and the corresponding domain document collection.

Output
Ranked documents
An ordered list of document identifiers for the complete query.

## Sub-track 1b — Step-wise Temporal Retrieval
Input
Decomposed steps + corpus
The temporal query is represented through reasoning steps, each targeting a necessary part or period.

Output
Rankings by step
An ordered document list for each intermediate retrieval step.

https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027

hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset --local-dir reteco_data