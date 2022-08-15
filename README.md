realtime_dreamer
==============================

This Realtime Dreamer project is performing NLP for Vietnamese language and machine learning tasks to settle a real-world business challenge of a company in Vietnam . We split our project into below four tasks:

1.	Customer review type classification

Suwasit Wittayaijug used customer reviews to train Phobert pre-trained BERT model and then classify review types such as delivery, product, quality, and service. Performed model evaluation and produced visualizations.

2.	Sentiment analysis for customer reviews

Yunhong He selected positive and negative key word to search the customer reviews and label them after eye scan. And then used these selected reviews with labels to train trituenhantaoio/bert-base-vietnamese-uncased pre-trained BERT model, and label the customer reviews with emotions such as positive, neutral and negative. Performed model evaluation and visualizations for 3 different pre-trained BERT models such as trituenhantaoio, Phobert and bert-base-uncased, as well as Supervised Machine Learning.  Setup team GitHub with folder structures. Produced sentiment analysis Machine Learning pipeline.

3.	Recommendation system

Kensuke Suzuki used user id, product and customer rating to train Memory-Based Collaborative Filtering model which will then recommend items to the user. “Users who are similar to you also liked…” Conducted model evaluation.

4.	Machine Learning for Sales Prediction

Zheng Wei Lim used E-Commerce data train Supervised ML algorithms, conducted feature engineering. Performed model evaluation of multiple ML algorithms, and produce visualizations. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    |   |   └── sentiment_analysis_reivew_emotion_predition.xlsx
    |   |   └── sentiment_analysis_trituenhantaoio_train_data_from_Yunhong He_Epoch1_accuracy_per_class_df.csv
    |   |   └── sentiment_analysis_trituenhantaoio_Epoch1_train_data_from_Yunhong He_eval_df.csv
    |   |   └── sentiment_analysis_best_bert_model.model
    |   |   └── sentiment_analysis_trituenhantaoio_train_data_provided_by_Yunhong He_NLP_Epoch10.model
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   │   └── Git_mockup_reviews_processed.xlsx
    │   │   └── sentiment_analysis_emotion_trituenhantaoio_1.model
    │   │   └── sentiment_analysis_reviews_label.xlsx
    │   │   └── sentiment_analysis_reviews_label_processed.csv
    |   |   └── sentiment_analysis_reviews_label_split.csv
    │   └── raw            <- The original, immutable data dump.
    │       └── Git_mockup_reviews.xlsx
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    |       └── sentiment analysis - model evaluation by individual class prediction accuracy - visualization.png
    |       └── sentiment analysis - model evaluation by metrics - visualization.png
    |       └── sentiment analysis - model evaluation by pre-trained bert model - visualization.png
    |       └── sentiment analysis - supervised ml model evaluation - visualization
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    |── sentiment_analysis.sh
    |
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── sentiment_analysis_data_utility_functions.py
    |   |   └── sentiment_analysis_predict_emotion.py
    |   |   └── sentiment_analysis_prepare_review_label.py
    |   |   └── sentiment_analysis_utility_functions.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── sentiment_analysis_train_bert_model.py
    │   │   └── sentiment_analysis_utility_functions.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
