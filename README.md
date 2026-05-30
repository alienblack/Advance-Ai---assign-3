# a1974524 Assignment 3: Kubernetes Security RAG

This folder contains the final notebook report and the helper code used to build, evaluate, and analyse a retrieval-augmented generation system for Kubernetes security hardening.

Repository target: `https://github.com/alienblack/Advance-Ai---assign-3`

## Main files

- `a1974524_A3.ipynb`: final report notebook
- `src/phase1_pipeline.py`: Phase 1 ingestion, parsing, chunking, embedding, and snapshot helpers
- `src/phase2/`: Phase 2 retrieval, generation, evaluation, analysis, and reporting helpers
- `src/project_paths.py`: project-root and output-path resolution helpers for local and Colab runs
- `requirements_colab.txt`: pinned package list used in the Colab setup cell
- `output/`: generated figures, tables, and saved run summaries

## Folder layout

```text
project-root/
├── a1974524_A3.ipynb
├── src/
├── data/
├── output/
├── requirements_colab.txt
└── README.md
```

## Reproducibility design

The notebook is set up so that the project root does not need to be edited by hand every time.

It resolves the project folder in this order:

1. `A3_PROJECT_ROOT` environment variable, if set
2. the current working directory if it already looks like the project root
3. nearby folders such as `a1974524_a3`, `a1974524_a3 2`, or `Advance-AI-Assign-3/a1974524_a3`
4. the default Colab Drive location `/content/drive/MyDrive/Advance-AI-Assign-3/a1974524_a3`

If auto-detection fails, set:

```bash
export A3_PROJECT_ROOT="/absolute/path/to/a1974524_a3"
```

In normal use, no path change is needed inside the notebook code. The current downloaded copy can still be named `a1974524_a3 2`; if you move or rename the folder and auto-detection stops working, set `A3_PROJECT_ROOT` once before opening the notebook.

## Running locally

1. Place the folder anywhere on your machine.
2. Open a terminal in the project root.
3. Create a Python environment if needed.
4. Install the pinned packages from `requirements_colab.txt` or install the notebook cell equivalents.
5. Launch Jupyter and open `a1974524_A3.ipynb`.
6. Run the notebook from the top.

Optional local override:

```bash
export A3_PROJECT_ROOT="/absolute/path/to/a1974524_a3"
```

## Running in Colab

1. Upload or sync the whole folder into Google Drive.
2. The default expected location is:

```text
/content/drive/MyDrive/Advance-AI-Assign-3/a1974524_a3
```

3. Open `a1974524_A3.ipynb` in Colab.
4. Run the package setup cell first.
5. Run the Drive mount and environment setup cell.
6. Run the notebook from top to bottom.

Required Colab secrets for the optional API-based sections:

- `HF_TOKEN`
- `OPENAI_API_KEY`

The notebook still runs without those secrets for the core local retrieval, generation, and proxy-based evaluation sections.

## Files that already support the final report

The final notebook expects these saved qualitative files to be present if you want to reproduce the later sections exactly:

- `output/phase2/tables/manual_scoring_100_filled.csv`
- `output/phase2/tables/failure_taxonomy_100_filled.csv`
- `output/phase2/tables/deep_dive_cases.csv`

These are already included in this project copy.

## Which sections regenerate outputs

- Phase 1 writes snapshots, normalised documents, chunks, embeddings, and Chroma artifacts into `data/` and `output/`
- Phase 2 writes retrieval, generation, evaluation, and analysis outputs into `output/phase2/tables`
- The report section displays the saved figures from `output/figures` and saved CSVs from `output/phase2/tables`

## Important note on saved figures and tables

The report-style results section now loads the saved artifacts directly from the `output/` folder. This means:

- rerunning the expensive generation or evaluation cells is not required just to display the report
- the report view is consistent with the saved outputs already included in the folder

## If the notebook cannot find the project root

Use one of these fixes:

1. run the notebook from inside the project folder
2. move the folder so that the notebook sits directly inside the project root
3. set `A3_PROJECT_ROOT` to the correct absolute path

## Recommended files to commit

If you push this folder to GitHub, the important files are:

- `a1974524_A3.ipynb`
- `src/`
- `requirements_colab.txt`
- `README.md`
- the required saved outputs under `output/`

Large backups like `*.bak*` and Python cache files do not need to be committed.
