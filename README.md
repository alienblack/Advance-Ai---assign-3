# Submission-Ready Bundle

This folder contains the final files I plan to commit for **Phase 1: Ingestion, Normalization, Chunking, and Embedding**.

## Files Included

- `requirements.txt`
- `notebooks/01_ingestion_pipeline.ipynb`
- `docs/phase1_report_notes.md`
- `output/`

## What I Should Do Next

1. Open `notebooks/01_ingestion_pipeline.ipynb`.
2. Run the notebook from the top.
3. Let the notebook save the latest figures, tables, summaries, and `all_chunks.txt` into `output/`.
4. Commit this `submission_ready` folder after the run finishes.

## What the Notebook Does

The notebook runs the full Phase 1 pipeline:

1. scrape trusted live Kubernetes security sources
2. save raw snapshots for reproducibility
3. normalize HTML, PDF, and JSON feed content into one document schema
4. build sentence-aware chunks
5. generate dense embeddings with `all-MiniLM-L6-v2`
6. save a persistent Chroma vector store
7. save final figures, tables, and summary files into `output/`

## Output Files I Expect After Running

The notebook will save these main outputs:

- `output/all_chunks.txt`
- `output/phase1_summary.json`
- `output/phase1_summary.md`
- `output/phase1_run_summary.json`
- `output/embedding_summary.json`
- `output/chroma_summary.json`
- `output/figures/`
- `output/tables/`

## Important Note

The notebook will also create a working `data/` folder inside this submission bundle when it runs. That folder is useful for execution, but it is not meant to be committed. The local `.gitignore` file already excludes it.
