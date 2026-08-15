# Emotional Arc Analysis

This project maps the emotional trajectory of *Harry Potter and the Goblet of Fire* using
sentence-level natural language processing. The entire implementation and its saved visual
results are contained in `Project.ipynb`.

## Repository files

- `Project.ipynb` — complete analysis, functions, visualizations, and saved results.
- `Harry Potter And The Goblet Of Fire.txt` — source text analyzed by the notebook.
- `README.md` — project documentation.

## Analysis

The notebook:

- Validates and extracts all 37 chapters, including hyphenated chapter numbers.
- Applies VADER to original sentences so punctuation, capitalization, and negation remain intact.
- Produces a 25-sentence rolling emotional arc.
- Calculates chapter means, medians, quartiles, polarity shares, and emotional volatility.
- Measures sentiment around Harry, Ron, Hermione, and Voldemort mentions with neighboring context.
- Extracts the strongest positive and negative passages.
- Compares distinctive unigrams and bigrams using TF-IDF.
- Benchmarks TF-IDF against SBERT semantic embeddings with K-Means clustering.
- Supports optional transformer-based emotion classification.
- Supports evaluation against manually labeled positive, neutral, and negative sentences.

## TF-IDF versus SBERT result

The notebook divides the novel into 1,811 non-overlapping, chapter-bounded passages and tests
both representations across 2–8 clusters with identical K-Means settings and cosine silhouette
scoring.

- Best TF-IDF silhouette: `0.0128` at 7 clusters.
- Best SBERT silhouette: `0.0604` at 2 clusters.
- Absolute silhouette gain: `0.0476`.
- Relative silhouette gain: `371.06%` (`4.71x`).
- Mean five-seed cluster stability: `0.399` ARI for TF-IDF and `0.975` for SBERT.

The comparison uses `sentence-transformers/all-MiniLM-L6-v2`, 6,946-dimensional TF-IDF
features, and 384-dimensional SBERT embeddings. The complete experiment and comparison chart
are stored in the notebook.

## Running the project

Open `Project.ipynb` from the repository root and run its cells in order. Install the baseline
dependencies if needed:

```python
%pip install pandas numpy nltk matplotlib seaborn scikit-learn
```

The notebook creates a local `outputs/` directory when executed. Generated output files do not
need to be committed because the principal charts are already stored inside the notebook.

SBERT and transformer emotion analysis additionally require `sentence-transformers`,
`transformers`, and `torch`. Their first run may download model weights.
