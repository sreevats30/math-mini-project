# Data

## news_400.csv

400 short news articles, 100 from each of the four AG News sections (World, Sports, Business, Sci/Tech).

- **Source:** the AG News test split (7,600 articles) released with Zhang, Zhao and LeCun (2015), *Character-level Convolutional Networks for Text Classification*, built from AG's corpus of news articles. AG's corpus is provided for research and non-commercial use.
- **Cleaning:** HTML entities decoded (`#36;` → `$`), backslash line breaks removed, wire-agency tags such as `(AP)` and `LONDON (Reuters) -` stripped so that the source agency cannot act as a topic signal.
- **Sampling:** only articles whose description has 22–60 words were kept, then 100 per section were drawn at random with seed 7.
- **Columns:** `id`, `category` (the AG News section), `title`, `text`. The app analyses `title + ". " + text`.

The labels are only used to score clustering after the fact. They never influence the term–document matrix or the SVD.

## science_24.json

The 24 one-sentence documents (AI, space, medicine, finance) written for the first version of this project.

## The classic example

The 9 titles from Deerwester et al. (1990), Table 2, are defined in `lsa_lab/corpora.py`, together with the paper's 12 index terms (the words that occur in at least two titles).
