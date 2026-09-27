import re
import gc
from html import unescape
import numpy as np
import pandas as pd
import spacy
from gensim.corpora import Dictionary
from gensim.models.phrases import Phrases, Phraser
from gensim import matutils
from sklearn.decomposition import LatentDirichletAllocation
from tqdm.auto import tqdm



# config
N_TOPICS = 12
RANDOM_STATE = 42

PHRASE_MIN_COUNT = 10
PHRASE_THRESHOLD = 10

VOCAB_NO_BELOW = 5
VOCAB_NO_ABOVE = 0.40
LDA_MAX_FEATURES = None

LDA_BATCH_SIZE = 2048
LDA_MAX_ITER = 5

SPACY_BATCH_SIZE = 1000
SPACY_N_PROCESS = -1


# stopwords to ignore

CUSTOM_STOPWORDS = {
    "thing", "stuff", "item", "something", "anything", "product", "products",
    "use", "used", "using", "get", "got", "make", "makes", "made",
    "work", "works", "tried", "try", "really", "just", "also", "much",
    "one", "time", "way", "little", "don", "doesn", "didn", "isn",
    "wasn", "weren", "wouldn", "couldn", "shouldn", "hasn", "haven",
    "aren", "purchase", "purchased", "purchasing", "friend", "family",
    "especially", "want", "wanted", "think", "thought", "know", "knew",
    "like",
}



# functions for preprocessing the reviews

def _to_text_list(texts):
    if isinstance(texts, pd.Series):
        return texts.fillna("").astype(str).tolist()

    result = []
    for x in texts:
        if x is None or (not isinstance(x, (list, dict, tuple, set)) and pd.isna(x)):
            result.append("")
        else:
            result.append(str(x))
    return result

def clean_html(text):
    text = unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def basic_clean(text):
    text = clean_html(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def process_docs(texts, nlp, stop_words, batch_size=1000, n_process=-1, desc="spaCy preprocessing"):
    docs = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)

    processed = []
    for doc in tqdm(docs, total=len(texts), desc=desc):
        tokens = []
        for token in doc:
            lemma = token.lemma_.lower()
            if token.is_alpha and len(lemma) > 2 and lemma not in stop_words:
                tokens.append(lemma)
        processed.append(tokens)

    return processed

def corpus_to_csr(corpus, n_terms):
    return matutils.corpus2csc(
        corpus,
        num_terms=n_terms,
        dtype=np.float32
    ).T.tocsr()


def print_topics(lda_model, dictionary, n_top_words=15):
    feature_names = np.asarray([dictionary[i] for i in range(len(dictionary))])

    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        words = feature_names[top_indices]
        print(f"Topic {topic_idx + 1:2d}: " + ", ".join(words))


# main
def run_lda(
    train_text,
    all_text,
    n_topics=N_TOPICS,
    random_state=RANDOM_STATE,
    phrase_min_count=PHRASE_MIN_COUNT,
    phrase_threshold=PHRASE_THRESHOLD,
    vocab_no_below=VOCAB_NO_BELOW,
    vocab_no_above=VOCAB_NO_ABOVE,
    max_features=LDA_MAX_FEATURES,
    lda_batch_size=LDA_BATCH_SIZE,
    lda_max_iter=LDA_MAX_ITER,
    spacy_batch_size=SPACY_BATCH_SIZE,
    spacy_n_process=SPACY_N_PROCESS,
    show_topics=True,
    n_top_words=10,
    return_pipeline=False,
):

    train_text = _to_text_list(train_text)
    all_text = _to_text_list(all_text)

    if len(train_text) == 0:
        raise ValueError("train_text is empty.")

    if len(all_text) == 0:
        raise ValueError("all_text is empty.")

    print(f"Training reviews: {len(train_text):,}")
    print(f"Reviews to transform: {len(all_text):,}")
    print(f"LDA topics: {n_topics}")

    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError as e:
        raise OSError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "In Colab run: !python -m spacy download en_core_web_sm"
        ) from e

    stop_words = set(nlp.Defaults.stop_words) | CUSTOM_STOPWORDS

    # clean + preprocess text
    print("\nCleaning training text...")
    train_clean = [basic_clean(text) for text in train_text]

    print("Running spaCy on training text...")
    train_base_tokens = process_docs(
        train_clean,
        nlp,
        stop_words,
        batch_size=spacy_batch_size,
        n_process=spacy_n_process,
        desc="Training preprocessing"
    )

    del train_clean
    gc.collect()

    # Fit bigram + trigram models
    print("\nTraining bigram model...")
    bigram_model = Phrases(
        train_base_tokens,
        min_count=phrase_min_count,
        threshold=phrase_threshold
    )
    bigram = Phraser(bigram_model)

    train_bigram_tokens = [bigram[doc] for doc in train_base_tokens]

    print("Training trigram model...")
    trigram_model = Phrases(
        train_bigram_tokens,
        min_count=phrase_min_count,
        threshold=phrase_threshold
    )
    trigram = Phraser(trigram_model)

    train_tokens = [trigram[bigram[doc]] for doc in train_base_tokens]

    del train_bigram_tokens, train_base_tokens
    gc.collect()

    # fit dictionary on training text only
    print("\nBuilding training dictionary...")
    dictionary = Dictionary(train_tokens)

    print(f"Initial vocabulary size: {len(dictionary):,}")

    dictionary.filter_extremes(
        no_below=vocab_no_below,
        no_above=vocab_no_above,
        keep_n=max_features
    )

    print(f"Filtered vocabulary size: {len(dictionary):,}")

    if len(dictionary) == 0:
        raise ValueError(
            "LDA vocabulary is empty. Reduce vocab_no_below or increase vocab_no_above."
        )

    # Build corpus and fit LDA
    train_corpus = [dictionary.doc2bow(doc) for doc in train_tokens]

    train_corpus_nonempty = [doc for doc in train_corpus if len(doc) > 0]

    if len(train_corpus_nonempty) == 0:
        raise ValueError("No non-empty training documents remain after preprocessing.")

    tf_train = corpus_to_csr(train_corpus_nonempty, len(dictionary))

    print(
        f"\nLDA training matrix: "
        f"{tf_train.shape[0]:,} docs x {tf_train.shape[1]:,} words"
    )

    print(f"Fitting {n_topics}-topic LDA...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="online",
        batch_size=lda_batch_size,
        max_iter=lda_max_iter,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    lda_model.fit(tf_train)

    if show_topics:
        print("\n" + "=" * 80)
        print("FITTED TOPICS")
        print("=" * 80)
        print_topics(lda_model, dictionary, n_top_words=n_top_words)

    del train_tokens, train_corpus, train_corpus_nonempty, tf_train
    gc.collect()

    # preprocess all text now
    print("\nCleaning all reviews...")
    all_clean = [basic_clean(text) for text in all_text]

    print("Running spaCy on all reviews...")
    all_base_tokens = process_docs(
        all_clean,
        nlp,
        stop_words,
        batch_size=spacy_batch_size,
        n_process=spacy_n_process,
        desc="All-review preprocessing"
    )

    del all_clean
    gc.collect()

    # Apply TRAINING phrase models to ALL text
    print("\nApplying training phrase models to all reviews...")
    all_tokens = [trigram[bigram[doc]] for doc in all_base_tokens]

    del all_base_tokens
    gc.collect()

    # Transform now all reviews with training dictionary + LDA
    print("Building all-review document-term matrix...")
    all_corpus = [dictionary.doc2bow(doc) for doc in all_tokens]
    tf_all = corpus_to_csr(all_corpus, len(dictionary))

    print(f"Transform matrix: {tf_all.shape[0]:,} docs x {tf_all.shape[1]:,} words")
    print("Calculating topic probabilities...")

    lda_topics = lda_model.transform(tf_all)

    if lda_topics.shape[0] != len(all_text):
        raise RuntimeError(
            f"Output row mismatch: expected {len(all_text):,}, "
            f"got {lda_topics.shape[0]:,}."
        )

    print(f"\nDone. lda_topics shape: {lda_topics.shape}")

    pipeline = {
        "lda_model": lda_model,
        "dictionary": dictionary,
        "bigram": bigram,
        "trigram": trigram,
        "nlp": nlp,
        "stop_words": stop_words,
    }

    if return_pipeline:
        return lda_topics, pipeline

    return lda_topics
