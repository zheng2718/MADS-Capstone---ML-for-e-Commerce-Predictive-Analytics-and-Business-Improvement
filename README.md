University of Michigan | Master of Applied Data Science 
Capstone Project - ML for e-Commerce Predictive Analytics and Business Improvement
==============================
<img width="1391" alt="Screen Shot 2565-08-19 at 23 56 15" src="https://user-images.githubusercontent.com/100912986/185674568-bfac3364-6555-4d8c-88c0-79101a23269e.png">

This Realtime Dreamer project is to use real Ecommerce data to perform NLP for Vietnamese language and machine learning tasks in order to settle a real-world business challenge of a company in Vietnam. We split our project into four tasks:

For the full report, please visit our blog: https://madsrealtimedreamer.wordpress.com/

Capstone Video Link: https://drive.google.com/file/d/1hJkoycGnJdH8OE1k9IqpxXhdyVOSLkBK/view?usp=sharing

<h2>1.	Customer Reviews Type classification</h2>


<img width="426" alt="Screen Shot 2565-08-22 at 12 29 41" src="https://user-images.githubusercontent.com/100912986/185845938-8a8ada6f-1325-4300-b306-c572414ed7f9.png">


Our  goal is to build a NLP model to predict the review classes using the customer reviews (sample data: Git_mockup_reviews.xlsx). The data collection process was completed by the customer service team of the company, who manually collected this data from the E-commerce platform. They recorded the review sentences (in Vietnamese) and the true rating score, along with labeling the type of the review manually.

For this task, we adopted a model called PhoBERT, which is a BERT base program (Bidirectional Encoder Representations from Transformers) released in late 2018. The objective of using this model  is to compare the performance of  PhoBERT  NLP to other traditional algorithms.


We can conclude that we can use Phobert as a tokenizer and transform it to train the review data. Unlike the previous monolinguals and multilingual approaches, Phobert is superior in attaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of Dependency parsing, Named-entity recognition, Part-of-speech tagging, and Natural language inference. For this reason, it is the best algorithm to predict the reviews classification tasks because of its superiority compared to other algorithms. While the data imbalance was an issue due to moderate, we overcame it by over-sampling the minority. The outcome was optimal based on the elements of the task and no data preprocessing. The PhoBERT model requires parameter tuning, and from the results, we were able to increase batch_size to 24 and  dropout to 0.4 to be the best to handle overfitting.


<strong>  </strong><br/>

<strong>  </strong><br/>


<h2>2.	Sentiment Analysis For Online Customer Reviews</h2>

We conducted Sentiment Analysis for Online Customer Reviews and created interactive sentiment analysis dashboard visualizations for review emotions, review ratings and reviews content class. Some content of the online customer reviews does not reflect the ratings. For example, some with 5-rating reviews have negative comments and emotions such as positive, neutral, and negative. Sentiment analysis for Vietnamese reviews is needed for the company to better understand their customers’ needs in order to improve product and sales performance.

We created reviews label dataset, trained and evaluated 3 Hugging Face Pre-trained BERT models including trituenhantaoio/bert-base-vietnamese-uncased, NlpHUST/vibert4news-base-cased, and bert-base-uncased, as well as Supervised Machine Learning algorithms, produced model evaluation visualizations. Setup team GitHub with folder structures. Created sentiment analysis Deep Learning and Machine Learning pipeline. Run /realtime_dreamer/sentiment_analysis.sh.

<h2>3.	Recommendation system</h2>

We used user id, product, and customer rating to train the Memory-Based Collaborative Filtering model which will then recommend items to the user. “Users who are similar to you also liked…”. In our analysis, we determined that the company is not ready to implement a recommendation system at this time. We have decided to build out a recommendation system To show what an example of what an output looks like. For the example, we are showing the recommendations for User 52354.

To run the SVD algorithm for the recommender system, you will be required to install a python library called surprise: 

```pip install surprise```

After installing Surprise, run 
```recommendation_system.sh```
and the final output recommendation_for_user_52354.csv will be in the data/final folder. 


One of the limitations comes from the fact that this company does not natively provide an ID for each user. To overcome this, we made labels for users based on unique names and addresses. Even with this workaround, there have been questions about data integrity because there was a user that made approximately 1100 purchases, for context, the next user only made 24 purchases. This could skew the results because of how frequently that particular user shows up.


We have decided to build out the recommendation system using collaborative filtering to show how it could work within the data given. But before such a system is to be implemented within the organization, we recommend that they take these steps such as encouraging users to leave reviews to implement a more robust recommendation system.



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
    │   │   └── df__phobert_all.sav
    │   │   └── df__phobert_grouping.sav
    │   │   └── df_phobert_remove.sav
    │   │   └── df__phobert_all_upsamples.sav
    │   │   └── df_predict_all.sav
    │   │   └── df_predict_upsampling.sav
    │   │   └── df_predict_grouping.sav
    │   │   └── df_predict_remove.sav
    │   │   └── phobert1_eval_df_.csv
    │   │   └── phobert2_eval_df_.csv
    │   │   └── phobert3_eval_df_.csv
    │   │   └── phobert3_dropout_eval_df_.csv
    |   └── final               <- File with final data of the project
    │   │   └── reviewtype__accuracy_per_class_df.csv
    │   │   └── reviewtype_phobert_model.pt
    │   │   └── recommendation_for_user_52354.csv
    │   │ 
    │   │
    │   ├── interim             <- Intermediate data that has been transformed. Files with BERT model evaluation metrics.
    |   |   └── sentiment_analysis_reivew_emotion_predition.xlsx
    |   |   └── sentiment_analysis_trituenhantaoio_train_data_from_Yunhong He_Epoch1_accuracy_per_class_df.csv
    |   |   └── sentiment_analysis_trituenhantaoio_Epoch1_train_data_from_Yunhong He_eval_df.csv
    |   |   └── model_info.csv
    |   |   └── sentiment_analysis_trituenhantaoio_train_data_provided_by_Yunhong He_NLP_Epoch10_accuracy_per_class_df.csv
    |   |   └── sentiment_analysis_trituenhantaoio_train_data_from_YunhongHe_Epoch10_accuracy_per_class_BeforeOversample.csv
    |   |   └── sentiment_analysis_trituenhantaoio_train_data_provided_by_Suwasit_NLP_Epoch10_accuracy_per_class_df.csv
    |   |   └── sentiment_analysis_trituenhantaoio_NLP_Epoch10_train_data_provided_by_Yunhong He_eval_df.csv
    |   |   └── sentiment_analysis_trituenhantaoio_NLP_Epoch10_train_data_provided_by_Yunhong He_eval_before_oversample_df.csv
    |   |   └── sentiment_analysis_trituenhantaoio_NLP_Epoch10_train_data_provided_by_Suwasit_eval_df.csv
    |   |   └── sentiment_analysis_NlpHUST_train_data_provided_by_Yunhong He_NLP_Epoch10_accuracy_per_class_df.csv
    |   |   └── sentiment_analysis_NlpHUST_train_data_provided_by_Suwasit_NLP_Epoch10_accuracy_per_class_VnEmoLex_validated_df.csv
    |   |   └── sentiment_analysis_NlpHUST_train_data_provided_by_Suwasit_NLP_Epoch10_accuracy_per_class_df.csv
    |   |   └── sentiment_analysis_NlpHUST_NLP_Epoch10_train_data_provided_by_Yunhong He_eval_df.csv
    |   |   └── sentiment_analysis_NlpHUST_NLP_Epoch10_train_data_provided_by_Suwasit_eval_VnEmoLex_validated_df.csv
    |   |   └── sentiment_analysis_NlpHUST_NLP_Epoch10_train_data_provided_by_Suwasit_eval_df.csv
    |   |   └── sentiment_analysis_bert-base-uncased_train_data_provided_by_Yunhong He_NLP_Epoch10_accuracy_per_class_df.csv
    |   |   └── sentiment_analysis_bert-base-uncased_train_data_provided_by_Suwasit_NLP_Epoch10_accuracy_per_class_df.csv
    |   |   └── sentiment_analysis_bert-base-uncased_NLP_Epoch10_train_data_provided_by_Yunhong He_eval_df.csv
    |   |   └── sentiment_analysis_bert-base-uncased_NLP_Epoch10_train_data_provided_by_Suwasit_eval_df.csv
    |   |   └── reviewType_pho_bert_eval_df.csv
    |   |   
    |   |   
    |   |   
    │   ├── processed           <- The final, canonical data sets for modeling.
    │   │   └── Git_mockup_reviews_processed.xlsx
    │   │   └── sentiment_analysis_reviews_label.xlsx
    │   │   └── sentiment_analysis_reviews_label_processed.csv
    |   |   └── sentiment_analysis_reviews_label_split.csv
    |   |   └── reviewType_pre_process.csv
    |   |   └── reviewType_df_upload.csv
    |   |   └── reviews_with_user_id.csv
    |   |
    │   └── raw                 <- The original, immutable data dump.
    │       └── Git_mockup_reviews.xlsx
    │       └── Anon-Data-productperformance17months.csv 
    │       └── Anon-Data-Review-Emotion-Prediction.xlsx 
    │       └── Anon-Data-cleaned-shipping.csv
    | 
    |
    ├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models                  <- Trained and serialized models, model predictions, or model summaries 
    |   └── sentiment_analysis_best_bert_model.model  <- link for this model: 
    │   │   https://drive.google.com/file/d/1ndzGpSsbzQ5mYRXkPMmzzg6bJmLLlD3q/view?usp=sharing
    │   └── sentiment_analysis_trituenhantaoio_train_data_provided_by_Yunhong He_NLP_Epoch10.model  
    │   │    <- link: https://drive.google.com/file/d/1ffLZd2jr5CGxGweuBq2bcM6lzIca66JB/view?usp=sharing
    │   └── predictive-final-model.sav <- final XGBoost Random Forest model for sales prediction    
    |
    ├── notebooks               <- Jupyter notebooks. A naming convention is a number (for ordering),
    │   |                          the creator's name, and a short `-` delimited description, e.g.
    │   |                          `1.0-jqp-initial-data-exploration.
    |   └── Sentiment_Analysis_Supervised_Machine_Learning_colab.ipynb   
    |   |   <- Complete Reviews label dataset is run in the notebook which can be run in Google Colab.
    |   └── Sentiment_Analysis_Supervised_Machine_Learning_Model_Evaluation_local.ipynb 
    |   |   <- Complete Reviews label dataset is run in this notebook.
    |   └── Sentiment_Analysis_BERT_Model_Evaluation.ipynb 
    |   |   <- Complete Reviews label dataset is run in this notebook.
    |   └── Sentiment_Analysis_BERT_Model_Evaluation.zip  
    |   |   <- Using zip file of the notebook to preserve the visualizations in the notebook.
    |   └── Model-Shipping-ProductPerformance-ReviewEmotion.ipynb  
    |   |   <- Predictive modeling on all merged data for sales forecasting
    |   └── Analytics-Shipping-Review.ipynb  <- EDA and visualization on shipping and review data
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    |   └── Sentiment Analysis.docx  <- Sentiment Analysis report
    │   │
    │   └── figures             <- Generated graphics and figures to be used in reporting
    |       └── sentiment analysis - model evaluation by individual class prediction accuracy - visualization.png
    |       └── sentiment analysis - model evaluation by metrics - visualization.png
    |       └── sentiment analysis - model evaluation by pre-trained bert model - visualization.png
    |       └── sentiment analysis - supervised ml model evaluation - visualization.png
    |       └── sentiment analysis - imbalanced classes before oversampling - visualization.png
    |       └── sentiment analysis - models.png
    |       └── sentiment analysis - BERT Model training - Fine Tuning and
    |       └── sentiment analysis - Data preprocess steps.png
    |       └── sentiment analysis - Emotions vs rating.png
    |       └── sentiment analysis - Supervised Machine Learning Process.png
    |       └── sentiment analysis - emotions vs content class dashboard.png
    |       └── Sentiment analysis - process.png
    │       └── sentiment analysis - imbalanced classes before oversampling - visualization.png
    |       └── sentiment analysis - emotions vs higher ratings 4 and 5.png
    |       └── sentiment analysis - tasks.png
    |       └── reviewType_model_compare_traditional.png
    |       └── reviewType_model_compare.png
    |       └── confusion_phobert.png
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
    │   │   └── sentiment_analysis_data_utility_functions.py
    |   |   └── sentiment_analysis_predict_emotion.py
    |   |   └── sentiment_analysis_prepare_review_label.py
    |   |   └── sentiment_analysis_utility_functions.py
    |   |   └── reviewtype__prepare_review_label.py
    |   |   └── reviewtype__train_test_val_split.py
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
    │       └── reviewtype_chart1_tuning.py
    |       └── reviewtype_chart2_compare_traditional.py
    │
    └── tox.ini                 <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
