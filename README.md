# NLP

Jupyter notebooks I wrote while learning Natural Language Processing. They go from basic text preprocessing to word embeddings, and end with two text classification projects.

## Contents

### Text preprocessing

| Notebook | What it covers |
|---|---|
| [tokenization.ipynb](tokenization.ipynb) | Splitting text into sentences and words: `sent_tokenize`, `word_tokenize`, `wordpunct_tokenize`, `TreebankWordTokenizer` |
| [stemming.ipynb](stemming.ipynb) | Reducing words to their stems with the Porter, Regexp and Snowball stemmers (Snowball gives the best results) |
| [lemmatization.ipynb](lemmatization.ipynb) | Reducing words to dictionary form with `WordNetLemmatizer` |
| [stopwords.ipynb](stopwords.ipynb) | Removing common English stopwords, then stemming or lemmatizing what's left |
| [partsOfSpeechTag.ipynb](partsOfSpeechTag.ipynb) | Tagging each word with its part of speech using `nltk.pos_tag` |
| [NamedEntitiyRecog.ipynb](NamedEntitiyRecog.ipynb) | Finding named entities (people, organizations, places) with `nltk.ne_chunk` |

### Turning text into vectors

| Notebook | What it covers |
|---|---|
| [BagOfWords.ipynb](BagOfWords.ipynb) | Bag of Words with `CountVectorizer` on the SMS spam dataset |
| [N-Grams.ipynb](N-Grams.ipynb) | Bigrams and trigrams with `CountVectorizer(ngram_range=...)` |
| [Tf-idf.ipynb](Tf-idf.ipynb) | TF-IDF features with `TfidfVectorizer` |
| [word2vec.ipynb](word2vec.ipynb) | Word embeddings with Gensim, using the pretrained `word2vec-google-news-300` model |

### Projects

**Spam/ham classification** uses [SMSSpamCollection.csv](SMSSpamCollection.csv), a dataset of about 5,500 SMS messages each labeled `ham` or `spam`.

| Notebook | Approach | Test accuracy |
|---|---|---|
| [SpamHamProject.ipynb](SpamHamProject.ipynb) | Bag of Words + Multinomial Naive Bayes | 98.3% |
| [SpamHamProject.ipynb](SpamHamProject.ipynb) | TF-IDF + Multinomial Naive Bayes | 97.5% |
| [SpamHamWord2Vec.ipynb](SpamHamWord2Vec.ipynb) | Averaged Word2Vec vectors + Random Forest | 97.5% |

**Kindle review sentiment analysis** is in [KIndle Review Sentiment Analysis/](KIndle%20Review%20Sentiment%20Analysis/). It uses `all_kindle_review.csv`, which has 12,000 Amazon Kindle reviews rated 1 to 5. Ratings 1–2 are labeled negative (0) and ratings 3–5 positive (1).

The notebook [kindle_review.ipynb](KIndle%20Review%20Sentiment%20Analysis/kindle_review.ipynb) goes through these steps:
1. Cleaning: lowercasing, removing HTML, URLs and special characters, and removing stopwords while keeping negations such as "not"
2. Lemmatization
3. A stratified 80/20 train/test split
4. Bag of Words and TF-IDF features
5. Multinomial Naive Bayes

| Approach | Test accuracy |
|---|---|
| Bag of Words + Multinomial Naive Bayes | 84.8% |
| TF-IDF + Multinomial Naive Bayes | 69.7% |

The TF-IDF model labels most negative reviews as positive because there are twice as many positive reviews as negative ones. Word2Vec is next on the list for this project.

## Setup

```bash
pip install nltk scikit-learn pandas numpy gensim beautifulsoup4 lxml tqdm jupyter
```

The notebooks download the NLTK data they need with `nltk.download(...)`. That includes `punkt`, `stopwords`, `wordnet`, `averaged_perceptron_tagger`, `maxent_ne_chunker` and `words`.

The pretrained Google News Word2Vec model is about 1.6 GB. Gensim downloads it the first time you run `api.load('word2vec-google-news-300')`.

## Libraries

- [NLTK](https://www.nltk.org/): tokenization, stemming, lemmatization, stopwords, POS tagging and NER
- [scikit-learn](https://scikit-learn.org/): vectorizers, train/test split, classifiers and metrics
- [Gensim](https://radimrehurek.com/gensim/): Word2Vec
- [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/): working with the data
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/): removing HTML tags
