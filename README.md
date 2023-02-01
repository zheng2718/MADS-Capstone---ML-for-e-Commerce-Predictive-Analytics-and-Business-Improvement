University of Michigan | Master of Applied Data Science 
Capstone Project - ML for e-Commerce Predictive Analytics and Business Improvement
==============================
<img width="1391" alt="Screen Shot 2565-08-19 at 23 56 15" src="https://user-images.githubusercontent.com/100912986/185674568-bfac3364-6555-4d8c-88c0-79101a23269e.png">

This project uses real Ecommerce data to perform NLP for Vietnamese language and machine learning tasks in order to settle a real-world business challenge of a company in Vietnam. We split our project into four tasks: NLP, sentiment analysis, recommendation system, and machine learning for sales predictive modeling. 

For the full report, please visit our blog: https://madsrealtimedreamer.wordpress.com/

Capstone Video Link: https://drive.google.com/file/d/1hJkoycGnJdH8OE1k9IqpxXhdyVOSLkBK/view?usp=sharing

<h2>1.	Customer Reviews Type classification</h2>


<img width="426" alt="Screen Shot 2565-08-22 at 12 29 41" src="https://user-images.githubusercontent.com/100912986/185845938-8a8ada6f-1325-4300-b306-c572414ed7f9.png">


Our goal is to build a NLP model to predict the review classes using the customer reviews (sample data: Git_mockup_reviews.xlsx). The data collection process was completed by the customer service team of the company, who manually collected this data from the E-commerce platform. They recorded the review sentences (in Vietnamese) and the true rating score, along with labeling the type of the review manually.

For this task, we adopted a model called PhoBERT, which is a BERT base program (Bidirectional Encoder Representations from Transformers) released in late 2018. The objective of using this model  is to compare the performance of  PhoBERT  NLP to other traditional algorithms.

We can conclude that we can use Phobert as a tokenizer and transform it to train the review data. Unlike the previous monolinguals and multilingual approaches, Phobert is superior in attaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of Dependency parsing, Named-entity recognition, Part-of-speech tagging, and Natural language inference. 


<strong>  </strong><br/>


<h2>2.	Sentiment Analysis For Online Customer Reviews</h2>

We conducted Sentiment Analysis for Online Customer Reviews and created interactive sentiment analysis dashboard visualizations for review emotions, review ratings and reviews content class. Some content of the online customer reviews does not reflect the ratings. For example, some with 5-rating reviews have negative comments and emotions such as positive, neutral, and negative. Sentiment analysis for Vietnamese reviews is needed for the company to better understand their customers’ needs in order to improve product and sales performance.

We created reviews label dataset, trained and evaluated 3 Hugging Face Pre-trained BERT models including trituenhantaoio/bert-base-vietnamese-uncased, NlpHUST/vibert4news-base-cased, and bert-base-uncased, as well as Supervised Machine Learning algorithms, produced model evaluation visualizations. Setup team GitHub with folder structures. Created sentiment analysis Deep Learning and Machine Learning pipeline. Run /realtime_dreamer/sentiment_analysis.sh.

<h2>3.	Recommendation system</h2>

We used the customer and product data to train the Memory-Based Collaborative Filtering model which will then recommend items to the user. “Users who are similar to you also liked…”. We have decided to build out a recommendation system to show what an example of what an output looks like. Therefore, we have decided to build it out using collaborative filtering to show how it could work within the data given. But before such a system is to be implemented within the organization, we recommend that they take these steps such as encouraging users to leave reviews to implement a more robust recommendation system.



<h2>4.	Machine Learning for Sales Predictive Modeling</h2>
We used data of e-Commerce sales, customer, and product data to perform EDA and predictive modeling through feature engineering and selection, model development and evaluation of multiple supervised ML algorithms, unsupervised optimizations, hyperparameter tuning, and provided visualizations to avoid black box models. 

The final model used is a XGBoost Random Forest model with R2 score of 0.95 and RMSE of 88.5m - an improvement of 75.0% over the baseline model. This indicates that the model has high accuracy of predicting future revenue from past sales data, enabling the company to be more responsive and confident in its revenue forecasting. 

Predictive modeling notebook: https://github.com/yunhonghe/realtime_dreamer/blob/main/notebooks/Model-Shipping-ProductPerformance-ReviewEmotion.ipynb

EDA notebook: https://github.com/yunhonghe/realtime_dreamer/blob/main/notebooks/Analytics-Shipping-Review.ipynb

<img width="500" height="200" alt="Predictive modeling - Feature Engineering and Selection" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Predictive-modeling-viz-featurecorrelation.jpg">
Feature Engineering and Selection

<img width="400" height="200" alt="Predictive modeling - Model Development and Evaluation" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Predictive-modeling-viz-modelevaluation.jpg">
Model Development and Evaluation

<img width="500" height="200" alt="Predictive modeling - Hyperparameter Tuning" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Predictive-modeling-final-model-optuna-optimizationhistory.jpg">
Hyperparameter Tuning


--------------------
Project Organization
--------------------

    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    ├── README.md               <- The top-level README for developers using this project.
    ├── data
    │   ├── external            <- Data from third party sources.
    |   └── final               <- File with final data of the project
    │   │   └── reviewtype__accuracy_per_class_df.csv
    │   │   └── reviewtype_phobert_model.pt
    │   │   └── recommendation_for_user_52354.csv
    │   │ 
    │   │
    │   ├── interim             <- Intermediate data that has been transformed. Files with BERT model evaluation metrics.
    |   |   
    │   ├── processed           <- The final, canonical data sets for modeling.
    |   |
    │   └── raw                 <- The original, immutable data dump.
    |
    ├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries 
    |   └── sentiment_analysis_best_bert_model.model  <- link for this model: 
    │   │   
    │   └── sentiment_analysis_trituenhantaoio_train_data_provided_by_Yunhong He_NLP_Epoch10.model  
    │   │    
    │   └── predictive-final-model.sav <- final XGBoost Random Forest model for sales prediction    
    |
    ├── notebooks              
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    |   └── Sentiment Analysis.docx  <- Sentiment Analysis report
    |
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`
    │
    |── sentiment_analysis.sh      <- The bash file to run the sentiment_analysis pipeline.
    |── recommendation_system.sh   <- The bash file to run the recommendation_system pipeline.
    |
    ├── setup.py                <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   │
    │   ├── data                <- Scripts to download or generate data
    │   │
    │   ├── features            <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models              <- Scripts to train models and then use trained models to make
    │   │   │                      predictions
    │   │   ├── sentiment_analysis_train_bert_model.py 
    │   │   └── sentiment_analysis_utility_functions.py
    │   │   └── reviewtype_train_phobert.py    
    │   │   └── reviewtype_validate_phobert_model.py    
    │   │   └── reviewtype_text_test_prediction.py
    │   │   └── recommendation_system.py
    │   │
    │   └── visualization       <- Scripts to create exploratory and results-oriented visualizations
    └── tox.ini                 <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
