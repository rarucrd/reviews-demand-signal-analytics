This repo contains the code for my master thesis "Enhancing E-Commerce Demand Forecasting: A Comparative Study of Classical NLP and LLM-Derived Consumer Signals"

The notebooks in this repo are set up in such a way that they can be run from Google Colab. Specifically, when running the local LLM, the compute with GPU must be selected.

One prerequisite is that the data must be downloaded to your drive (or file system if running from there). This can be obtained from https://amazon-reviews-2023.github.io/. The "Beauty_and_Personal_Care" dataset should be downloaded from there.

There are 4 files:

_**data_extract.ipynb**_ contains the code to extract the reviews and select a number of reviews and time range which to continue with.

_**main.ipynb**_ contains the code that takes the data, processes it, creates the features for the different data-tiers (including through using the LLM and LDA), and finally trains and evaluates all the model-data combinations.

_**LDA_exploring.ipynb**_ is used to determine the optimal amount of topics to use for LDA. This is done by taking the training data from an intermediate step in main.ipynb and calculating various metrics for several amounts of topics.

**_run_lda.py_** contains a function that is used within main.ipynb for preprocessing data, and subsequently fitting and scoring the data with LDA using the optimal amount of topics as determined through LDA_exploring.ipynb

