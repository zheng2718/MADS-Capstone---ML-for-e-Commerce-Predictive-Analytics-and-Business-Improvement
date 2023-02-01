University of Michigan | Master of Applied Data Science 
Capstone Project - ML for e-Commerce Predictive Analytics and Business Improvement
==============================
<img width="1391" alt="Screen Shot 2565-08-19 at 23 56 15" src="https://user-images.githubusercontent.com/100912986/185674568-bfac3364-6555-4d8c-88c0-79101a23269e.png">

This project uses real Ecommerce data to perform NLP for Vietnamese language and machine learning tasks in order to settle a real-world business challenge of a company in Vietnam. We split our project into four tasks: NLP, sentiment analysis, recommendation system, and machine learning for sales predictive modeling. 

<h2>Machine Learning for Sales Predictive Modeling</h2>
We used data of e-Commerce sales, customer, and product data to perform EDA and predictive modeling through feature engineering and selection, model development and evaluation of multiple supervised ML algorithms, unsupervised optimizations, hyperparameter tuning, and provided visualizations to avoid black box models. 

Exploratory data analysis (EDA) is conducted after the data is processed through an extract-transform-load (ETL) pipeline. The analysis revealed several area of weaknesses. As an example, one unexpected relationship is that there is little correlation between the ratings and factors such as shipping days. Further visualizations have been developed for the company, such as one showing that certain regions have more negative ratings, which indicates localized shipping issues to feed into on-the-ground investigations. 

Next, a ML model is built to allow the business to predict future behaviors of how products are likely to perform based on historical sales metrics. To develop accurate and reliable models, features are prepared, structured and selected that best describes the structure in the data. Additional features are engineered based on the E-commerce sales funnel.

<img width="500" height="200" alt="Predictive modeling - Feature Engineering and Selection" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Predictive-modeling-viz-featurecorrelation.jpg">
Feature Engineering and Selection

Several models are developed using supervised learning algorithms. Random forest is used as a baseline, along with more sophisticated models including regression and XGBoost random forest, which is an optimized distributed gradient boosting library.

<img width="400" height="200" alt="Predictive modeling - Model Development and Evaluation" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Predictive-modeling-viz-modelevaluation.jpg">

Hyperparameter tuning is also performed to enhance the models using Optuna. One hundred trials of the model are conducted on a search space for the optimal hyperparameters, allowing us to minimize the RMSE. 

The final model used is a XGBoost Random Forest model with R2 score of 0.95 and RMSE of 88.5m - an improvement of 75.0% over the baseline model. This indicates that the model has high accuracy of predicting future revenue from past sales data, enabling the company to be more responsive and confident in its revenue forecasting. 

<img width="500" height="200" alt="Predictive modeling - Hyperparameter Tuning" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Predictive-modeling-final-model-optuna-optimizationhistory.jpg">

Predictive modeling notebook: https://github.com/zheng2718/MADS-Capstone---ML-for-e-Commerce-Predictive-Analytics-and-Business-Improvement/blob/main/notebooks/Model-Shipping-ProductPerformance-ReviewEmotion.ipynb

EDA notebook: https://github.com/zheng2718/MADS-Capstone---ML-for-e-Commerce-Predictive-Analytics-and-Business-Improvement/blob/main/notebooks/Analytics-Shipping-Review.ipynb

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
