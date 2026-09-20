# JEV × Cosine Search Lab

日本語版は [README_JP.md](README_JP.md) にあります。

A local demo that compares cosine similarity retrieval with JEV's per-document
Noul judgment over the same 65 NordWind documents. Preset searches show correct
hits, false hits and missed documents against a fixed ground truth, together
with Precision, Recall and F1.

## Setup

Run these in any shell where Conda is available — an Anaconda Prompt or
PowerShell on Windows, Terminal on macOS or Linux. Every command below assumes
the cloned project folder — the one holding this README — as the working
directory. Replace `path/to/Jev-vs-VectorSearch` with wherever you put it, for
example `C:/GitHub/Jev-vs-VectorSearch` or `~/GitHub/Jev-vs-VectorSearch`.

```shell
cd path/to/Jev-vs-VectorSearch
conda create -n jev-search-demo python=3.10
conda activate jev-search-demo
python -m pip install -r requirements.txt
```

To reuse an existing Python 3.10 environment, activate it and run
`python -m pip install -r requirements.txt` there instead.

## API key

The JEV column needs a TypeSafe account and an API key. If you do not have one,
create an account at <https://typesafe.ai/> and issue an API key there.

Once you have the key, copy the bundled template to create the real key file.

```shell
cd path/to/Jev-vs-VectorSearch
cp .secrets/typesafe_api_key.example.txt .secrets/typesafe_api_key.txt
```

In an Anaconda Prompt on Windows, use
`copy .secrets\typesafe_api_key.example.txt .secrets\typesafe_api_key.txt`
instead; PowerShell understands `cp`.

Delete the comment lines in `.secrets/typesafe_api_key.txt` and paste the key
itself on the first line. Only the real key file is gitignored.

Without a key the app still starts and cosine search works; only the JEV column
reports that the key is missing.

## Interface language

The button in the top right switches the interface between Japanese and
English. The choice is stored in the browser and survives a reload. Parts that
are English to begin with — the search queries and the Cosine and JEV column
headings — stay as they are.

## Running

```shell
cd path/to/Jev-vs-VectorSearch
conda activate jev-search-demo
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The first cosine search takes a moment because it
downloads the embedding model and builds the cache.

## Using the app

1. Pick a preset with ground truth, or write your own query.
2. Choose whether JEV reads one document at a time or four in parallel.
3. Press Run.
4. Move each column's threshold and watch the cards and metrics change.

Editing a preset query turns it into a custom query, which disables ground-truth
evaluation. A JEV Noul score is the probability of "yes", while a cosine score
is a geometric similarity, so the two numbers are not meant to be compared
directly — only their outcomes against the ground truth are.

## Data

`dataset/documents.jsonl` is a snapshot copied from the NordWind workshop so
that this demo is self-contained. Nothing reads or writes the original
repository at runtime.
