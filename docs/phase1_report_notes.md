# Phase 1 Report Notes

These are my notes for writing the Phase 1 part of the report. This is not final report text. It is just what I need to cover.

## Main point of Phase 1

Phase 1 is about building the knowledge base properly before retrieval starts.

What I did in this phase:
- picked one focused cybersecurity topic: Kubernetes security hardening
- collected trusted live sources
- saved raw copies for reproducibility
- normalized different source types into one format
- chunked the text into retrieval units
- embedded the chunks
- stored the vectors in Chroma

Short idea I can use in the report:

`In Phase 1, I built a retrieval-ready cybersecurity corpus for Kubernetes security hardening using live trusted sources, structured preprocessing, sentence-aware chunking, dense embeddings, and a persistent vector store.`

## 1. Topic and source choice

What I need to say:
- I did not choose a broad topic like general phishing or all incident response.
- I narrowed the topic to Kubernetes security hardening so the knowledge base stays focused.
- I used trusted and domain-specific sources instead of random web search.

What sources are in the corpus:
- official Kubernetes documentation
- Kubernetes CVE feed
- NIST SP 800-190
- NSA/CISA hardening guidance
- OWASP Kubernetes Security Cheat Sheet
- CNCF whitepaper
- Aalto thesis
- practitioner books and whitepapers

Why this matters:
- the assignment asks for a specific cybersecurity topic
- the report should show that the corpus is not random
- this also helps later with relevance and faithfulness

Papers to cite here:
- CyberBOT
- CyberRAG

What I want to say from those papers:
- cybersecurity RAG systems usually work better with curated domain knowledge
- they do not rely on unrestricted web search for everything

## 2. Live scraping and reproducibility

What I need to say:
- my sources are live web pages, PDFs, and one JSON feed
- I fetched them from live URLs
- I also saved raw snapshots and metadata during the run

Why I did this:
- live sources can change
- I still need reproducibility for the assignment
- saving raw snapshots means I can explain exactly what was ingested in that run

Important point:
- this is not just scraping for the sake of scraping
- it is a controlled ingestion pipeline with an allowlist

Paper to cite here:
- RAG-Enhanced Large Language Model for Intelligent Assistance from Web-Scraped Data

What to use from it:
- web-scraped content can be turned into a RAG corpus
- scraped content can be embedded and indexed for retrieval

## 3. Normalization

What I need to say:
- my corpus is mixed: HTML, PDF, and JSON feed
- I could not send all of them directly into chunking because the structures are different
- I normalized everything into one common schema

Fields I preserved:
- source_id
- doc_id
- source_url
- section_path
- page_number when relevant
- text
- metadata like trust level and tags

Why this step matters:
- later retrieval should not depend on whether the original source was HTML or PDF
- normalization also helps traceability and later analysis

Simple way to explain it:

`Normalization converted different source types into one consistent document format so that chunking, embedding, and later retrieval could be applied uniformly across the corpus.`

## 4. Chunking

This is one of the main design choices, so I need to justify it properly.

What I used:
- sentence-aware chunking
- `max_words = 160`
- `overlap_words = 40`
- `min_words = 40`

Why I did not use naive splitting:
- fixed splitting can cut sentences at bad places
- that makes retrieval units weaker
- for RAG, chunk quality matters a lot

What I need to say about the final settings:
- `160` words gave a better balance than `120` or `200`
- `120 / 30 / 40` created more fragmentation
- `200 / 40 / 40` created larger chunks that were less focused
- reducing `min_words` to `20` increased the number of weak short chunks
- keeping `overlap = 40` was a safe way to preserve context at long boundaries

Numbers from my sensitivity table:
- `120 / 30 / 40` -> `2869` chunks, median `103` words
- `160 / 40 / 40` -> `2318` chunks, median `133` words
- `200 / 40 / 40` -> `1992` chunks, median `157` words
- `160 / 40 / 20` -> more short noisy chunks

Why I kept `160 / 40 / 40`:
- still above the assignment scale threshold
- chunks are more focused than the `200` setting
- less fragmented than the `120` setting
- better tradeoff overall

Papers to cite here:
- CyberBOT
- CyberRAG

What to use from them:
- semantically meaningful chunking is standard in RAG pipelines
- overlap is used to preserve context across boundaries

Extra point from my own notebook:
- I also did a qualitative chunk inspection, not just counts
- in the first `100` chunks, I judged `74` strong, `20` acceptable but generic, and `6` weak

That helps show that I actually checked chunk usefulness instead of only reporting statistics.

## 5. Scale

This has to be stated clearly because of the assignment announcement.

My final Phase 1 counts:
- `14` sources
- `38` fetched items
- `1381` normalized documents
- `2318` chunks

What I need to say:
- the dataset is not originally a Q/A dataset
- the assignment announcement allows chunks to count as answerable units
- my final corpus has `2318` chunks, so it clears the `2000` requirement

Simple line I can use:

`Since the assignment allows document chunks to be treated as answerable units when the corpus is not natively Q/A-formatted, the final corpus satisfies the large-scale requirement with 2318 retrieval chunks.`

## 6. Embeddings

What I embedded:
- final chunks, not raw full documents

Why:
- the chunk is the retrieval unit
- retrieval should happen over the same units I inspected and cleaned

Model I used:
- `all-MiniLM-L6-v2`

Why I picked it:
- lightweight and fast
- easy to run locally and in Colab
- common sentence-transformers baseline
- `384`-dimensional vectors
- good fit for short text and semantic similarity

What I should compare it against briefly:
- `multi-qa-mpnet-base-dot-v1`
- `bge-small-en`

My justification:
- I did not need a large heavy model for Phase 1
- I needed something reproducible and efficient
- for `2318` chunks, the lighter model is enough and easier to manage

Papers / sources to cite:
- SBERT
- MiniLM
- model card for `all-MiniLM-L6-v2`

## 7. Vector store

What I used:
- Chroma

Why:
- persistent local vector store
- stores vectors and metadata together
- easy to reuse in the next phase
- better pipeline story than only keeping a raw `.npy` file

What I should say:
- I also saved the embeddings separately
- Chroma was used as the actual retrieval-ready vector store

This helps with:
- reproducibility
- later retrieval experiments
- cleaner RAG pipeline design

## 8. Figures and analysis

I should mention the three main figures briefly.

### Normalized documents per source

Main point:
- the corpus is not evenly distributed
- the top 4 sources contribute `845 / 1381` normalized documents = `61.2%`

What this means:
- a few sources shape the corpus strongly
- but it is still not a single-source dataset

### Chunk word distribution

Main point:
- chunk sizes are mostly in the medium range
- median chunk length is `133` words

What this means:
- chunks are not mostly tiny fragments
- the chunker is behaving reasonably

### Chunks per source

Main point:
- top 4 chunk contributors = `1384 / 2318` = `59.7%`
- top 2 chunk contributors = `906 / 2318` = `39.1%`

What this means:
- corpus scale is good
- but source imbalance exists and I should mention it as a limitation

## 9. Limitations

I should keep the limitations simple and honest.

Main limitations:

1. Source imbalance
- two large book-style sources contribute `39.1%` of the final chunk set
- later retrieval may over-favor them compared with official guidance

2. Some weak chunks remain
- most chunks are usable
- but a small number are still short or generic

Good way to frame it:
- Phase 1 is strong overall
- but the corpus is not perfectly balanced and not perfectly clean

## 10. Which papers support which design choices

### CyberBOT
Use for:
- curated domain-specific corpus
- semantically meaningful chunking
- dense retrieval setup

### CyberRAG
Use for:
- overlap-aware chunking
- embeddings plus vector-store style retrieval
- cybersecurity-specific knowledge base design

### RAG-Enhanced LLM from Web-Scraped Data
Use for:
- scraping-based ingestion
- embedding and indexing scraped content

### RAGAS
Use mainly later in evaluation, but I can mention one thing now:
- coherent and source-linked chunks matter because later evaluation will look at retrieval relevance and faithfulness

## 11. Final closing note for Phase 1

If I need one short closing paragraph idea, the main message is:

- Phase 1 built a large-scale and retrieval-ready corpus
- the preprocessing is reproducible
- the chunking is justified
- the embedding and vector-store setup is ready for retrieval in Phase 2

Short version:

`Overall, Phase 1 gave me a reproducible and retrieval-ready Kubernetes security corpus. I collected trusted live sources, normalized them into one format, created sentence-aware chunks, embedded the final retrieval units, and stored them in a persistent vector database for the next phase.`
