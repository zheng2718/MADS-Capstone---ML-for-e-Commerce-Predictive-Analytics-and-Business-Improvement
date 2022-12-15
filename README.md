<h1>realtime_dreamer</h1>
==============================
<img width="1391" alt="Screen Shot 2565-08-19 at 23 56 15" src="https://user-images.githubusercontent.com/100912986/185674568-bfac3364-6555-4d8c-88c0-79101a23269e.png">

This Realtime Dreamer project is to use real Ecommerce data to perform NLP for Vietnamese language and machine learning tasks in order to settle a real-world business challenge of a company in Vietnam. We split our project into four tasks:

For the full report, please visit our blog: https://madsrealtimedreamer.wordpress.com/

Capstone Video Link: https://drive.google.com/file/d/1hJkoycGnJdH8OE1k9IqpxXhdyVOSLkBK/view?usp=sharing

<h2>1.	Customer Reviews Type classification</h2>


<img width="426" alt="Screen Shot 2565-08-22 at 12 29 41" src="https://user-images.githubusercontent.com/100912986/185845938-8a8ada6f-1325-4300-b306-c572414ed7f9.png">


Our  goal is to build a NLP model to predict the review classes using the customer reviews (sample data: Git_mockup_reviews.xlsx). The data collection process was completed by the customer service team of the company, who manually collected this data from the E-commerce platform. They recorded the review sentences (in Vietnamese) and the true rating score, along with labeling the type of the review manually

The pipeline script of this task created: reviewtype_script_.sh
 
 
<br/><strong><br/>Google Colab set up  and installation</strong><br/>
To use PhoBERT model, we need GPU resources. Google Colab offers free GPU. Since we will be training a large neural network, we will need to take full advantage of this.

GPUs can be added by going to the menu and selecting: Edit -> Notebook Settings -> Add accelerator ( GPU )

Then run the cell below to confirm that the GPU has been received.
 
    device = torch.device(“cuda:0” if torch.cuda.is_available() else “cpu”)


For this task, we adopted a model called PhoBERT, which is a BERT base program (Bidirectional Encoder Representations from Transformers) released in late 2018. The objective of using this model  is to compare the performance of  PhoBERT  NLP to other traditional algorithms.


To run the Phobert require installation  transformer and pytorch, so we have to install:



    !pip install transformer

    !pip install torch
    
<br/><strong>Data loading and preparation:</strong><br/>

It will load, clean, and up-sample data to handle the two minority classes imbalanced. This process will take raw Reviews data  (stored in /data/raw/Git_mockup_review.xlsx. )


And output reviewType_pre_process.csv
 
 
<br/><strong>Train_test_split the data </strong><br/>

We split data to Train and Test set , with test size =0.2 and then we split the Train data again to train_df and Val_df for model training and model loss validation, the process will take input of reviewType_pre_process.csv  and output reviewType_df_upload.csv ready to transmit to the dataloader function.
 
 
<br/><strong>Train the Phobert model</strong><br/>
reviewtype__train_test_val_split.py in src/data/ will  import PhoBERT's word separator (Tokenization) ,’vinai/phobert-base’ as below 
      
      
      from transformers import AutoModel, AutoTokenizer
      from transformers import BertForSequenceClassification

      phobert = AutoModel.from_pretrained(“vinai/phobert-base”).to(device)


Then we select Phobert  Vietnamese NLP as Tokenizer


      tokenizer = AutoTokenizer.from_pretrained(“vinai/phobert-base”)


 
which is used to convert the text into tokens corresponding to PhoBERT's lexicon. We load BertForSequenceClassification, which is a regular BERT model with a single linear layer added above for classification (Khanh, 2020). This process will be used to classify sentences. Once we provide the input data, the entire pre-trained BERT model and classifier will be trained with our specific task.



      model = BertForSequenceClassification.from_pretrained(
         'vinai/phobert-base', 
         num_labels = len(label_dict),
         output_attentions = False,
         output_hidden_states = False,
         hidden_dropout_prob= hidden_dropout_prob,
         attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
      model = model.to(device)


We set the training loop by asking the model to compute gradience and putting the model in training mode, then unpack our input data. Then we delete the gradience in the previous iteration, Backpropagation, and update the weight using optimize.step() then, by each,we save the best model which has the lowest validation loss.

 
The result of each training epoch will be saved in the reviewType_pho_bert_eval_df.csv in the ‘data/interim’ directory,
 
<strong><br/>Model Tuning comparison  </strong><br/>
We create reviewtype_chart1_tuning.py to compare the result of the best Phobert model with different hyperparameter and data process tuning, the other models have been trained from Colab environments and uploaded thier the Evaluation_df into ‘data/external’ directory; we are focusing on the F1 macro score and,F1 Average score and validation lost of each trained Epoch,  the result is shown in the ‘report/’ directory.

<img width="457" alt="Screen Shot 2565-08-22 at 18 42 58" src="https://user-images.githubusercontent.com/100912986/185913470-f8add998-32aa-4128-b2dc-6e8899188753.png"><img width="438" alt="Screen Shot 2565-08-22 at 18 43 11" src="https://user-images.githubusercontent.com/100912986/185913485-e5d5a002-e4e2-4211-a255-bbb7ee6e88e1.png">

 
The best model (number #4)  hyper parameter tuning recorded as below:
 
hidden_dropout_prob = 0.1

attention_probs_dropout_prob = 0.1<br/>
pre_trained_model = 'vinai/phobert-base'<br/>
model_type = pre_trained_model.split('/')[0]<br/>
batch_size = 8<br/>
epochs = 20<br/>
Ir = 1e-5<br/>
eps = 1e-8<br/>


 
<br/><strong>Comparing Phobert with other algorithms</strong><br/>
We also compare the result of all Phobert models with other traditional ML algorithms such as Random forest, SVM, and XGM classifier that are trained by using GridserchCv to tune hyperparameters. The best score from each parameter selected is imported to the ‘data/external’.

Again the best model is “Phobert- upsampling minority class, which able to provide F1 macro score at 0.90)

![reviewType_model_compare_traditional (1)](https://user-images.githubusercontent.com/100912986/185786652-38bb2353-2fe8-4791-903b-f47697751be3.png)

<strong><br/>Model Testing </strong><br/>

we have run  several manual Vietnamese sentense testing ,for example , we input text to the model, and predict a class.

Input_text='Bàn ủi hơi nước cầm tay tiện lợi Tefal - DT6130E0, hàng chính hãng bảo hành 2 năm'

Predict review type :  Quality


<strong>In conclusion</strong><br/>
We can conclude that we can use Phobert as a tokenizer and transform it to train the review data. Unlike the previous monolinguals and multilingual approaches, Phobert is superior in attaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of Dependency parsing, Named-entity recognition, Part-of-speech tagging, and Natural language inference. For this reason, it is the best algorithm to predict the reviews classification tasks because of its superiority compared to other algorithms. While the data imbalance was an issue due to moderate, we overcame it by over-sampling the minority. The outcome was optimal based on the elements of the task and no data preprocessing. The PhoBERT model requires parameter tuning, and from the results, we were able to increase batch_size to 24 and  dropout to 0.4 to be the best to handle overfitting.


<strong>  </strong><br/>

<strong>  </strong><br/>


<h2>2.	Sentiment Analysis For Online Customer Reviews</h2>

We conducted Sentiment Analysis for Online Customer Reviews and created interactive sentiment analysis dashboard visualizations for review emotions, review ratings and reviews content class. Some content of the online customer reviews does not reflect the ratings. For example, some with 5-rating reviews have negative comments and emotions such as positive, neutral, and negative. Sentiment analysis for Vietnamese reviews is needed for the company to better understand their customers’ needs in order to improve product and sales performance.

We created reviews label dataset, trained and evaluated 3 Hugging Face Pre-trained BERT models including trituenhantaoio/bert-base-vietnamese-uncased, NlpHUST/vibert4news-base-cased, and bert-base-uncased, as well as Supervised Machine Learning algorithms, produced model evaluation visualizations. Setup team GitHub with folder structures. Created sentiment analysis Deep Learning and Machine Learning pipeline. Run /realtime_dreamer/sentiment_analysis.sh.

<img width="400" alt="Sentiment Analysis Task" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20tasks.png">

Graph 1: Sentiment Analysis Task

<strong>  </strong><br/>

<img width="800" alt="Sentiment Analysis - Process" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Sentiment%20analysis%20-%20process.png">

Graph 2: Sentiment Analysis Process

<strong>  </strong><br/>

<strong>(1) Data used to train the model</strong><br/>

Yunhong He used clear positive and negative keyword search and eye scan to select customer reviews for emotion labeling. 

<img width="800" alt="Sentiment Analysis Data Preprocessing" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20Data%20preprocess%20steps.png">


Graph 3: Sentiment Analysis - Supervised Machine Learning Process

<strong>  </strong><br/>


Below graph shows that The classification performance of Supervised ML classifiers is very poor in terms of less than 0.4 F1 macro score for all the models. F1 micro and weighted scores are also lower than those in BERT models.

<img width="600" alt="Sentiment Analysis Supervised ML Classifier Model Evaluation" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20supervised%20ml%20model%20evaluation%20-%20visualization.png">




<strong>  </strong><br/>

<strong>(4) Other information</strong><br/>

(A) The zip file of Sentiment_Analysis_BERT_Model_Evaluation.ipynb can preserve the model evaluation visualizations and is located at https://github.com/yunhonghe/realtime_dreamer/blob/main/notebooks/Sentiment_Analysis_BERT_Model_Evaluation.7z

(B) The sentiment analysis pipeline is located at https://github.com/yunhonghe/realtime_dreamer. The https://github.com/yunhonghe/realtime_dreamer/blob/main/sentiment_analysis.sh is used to run the pipeline. 

(C) Below are the links for the sentiment analysis BERT models.

(a) The link to /models/sentiment_analysis_best_bert_model.model is https://drive.google.com/file/d/1ndzGpSsbzQ5mYRXkPMmzzg6bJmLLlD3q/view?usp=sharing

(b) The link to /models/sentiment_analysis_trituenhantaoio_train_data_provided_by_Yunhong He_NLP_Epoch10.model is  https://drive.google.com/file/d/1ffLZd2jr5CGxGweuBq2bcM6lzIca66JB/view?usp=sharing


(D) Sample files:

(a) Customer review label data in C:\Users\heyun\Capstone\realtime_dreamer\data\processed\sentiment_analysis_reviews_label.xlsx is the small sample of the file '/content/drive/MyDrive/Realtime Dreamer/train reviews.xlsx' used in notebooks\Sentiment_Analysis_BERT_Model_Evaluation.ipynb. Column "emotion" is labeled by Yunhong He using the keyword search method to select positive and negative customer reviews after an eye scan.

(b) Customer reviews dataset at C:\Users\heyun\Capstone\realtime_dreamer\data\processed\Git_mockup_reviews_processed.xlsx is the small sample of the customer review file 'drive/MyDrive/Realtime Dreamer/Tefal Lazada Product Reviews in TTL202207_Updated_Good_Bad.xlsx' used in notebooks\Sentiment_Analysis_BERT_Model_Evaluation.ipynb. Column "Comment classified Type 1" is labeled by the Vietnamese team.


(E) BERT model evaluation files:

Sentiment analysis model evaluation files are generated in Sentiment_Analysis_BERT_Model_Evaluation.ipynb and Sentiment_Analysis_BERT_Model_Evaluation.zip, and are used to produce BERT model evaluation visualizations, located at https://github.com/yunhonghe/realtime_dreamer/tree/main/data/interim




<h2>3.	Recommendation system</h2>

We used user id, product, and customer rating to train the Memory-Based Collaborative Filtering model which will then recommend items to the user. “Users who are similar to you also liked…”. In our analysis, we determined that the company is not ready to implement a recommendation system at this time. We have decided to build out a recommendation system To show what an example of what an output looks like. For the example, we are showing the recommendations for User 52354.

To run the SVD algorithm for the recommender system, you will be required to install a python library called surprise: 

```pip install surprise```

After installing Surprise, run 
```recommendation_system.sh```
and the final output recommendation_for_user_52354.csv will be in the data/final folder. 

There were a couple of roadblocks encountered during the analysis and as it stands today, we cannot implement the recommendation system. We will talk about limitations we have encountered and how we overcame them, and the limitations that make it impossible to implement the recommendation system.

One of the limitations comes from the fact that this company does not natively provide an ID for each user. To overcome this, we made labels for users based on unique names and addresses. Even with this workaround, there have been questions about data integrity because there was a user that made approximately 1100 purchases, for context, the next user only made 24 purchases. This could skew the results because of how frequently that particular user shows up.

Another limitation that we encountered was that the dataset was limited to kitchenware appliances, so this meant that our recommendations were limited to kitchenware appliances. This is important because, without the full basket of items, this narrows the accuracy of the recommendation system. While we were doing the analysis, we encountered a limitation that made it impossible for us to continue. It was found that there were multiple items associated with MODEL/DESC number. We had tried to distinguish by trying to find a sub-category, but the sub-category did not provide clarification. This is important because it makes it impossible to do a recommendation if multiple items are tied to a certain MODEL/DESC number.

We have decided to build out the recommendation system using collaborative filtering to show how it could work within the data given. But before such a system is to be implemented within the organization, we recommend that they take these steps before doing so:

•	Give each user an ID

•	Give each item a unique MODEL/DESC number

•	When preparing a dataset, make sure to include the full basket of items on the website

•	Encourage users to leave reviews

With these recommendations implemented, our team believes that the company will be able to implement a recommendation system.


<h2>4.	Machine Learning for Sales Predictive Modeling</h2>
We used data of E-Commerce sales, customer, and product data to perform EDA and predictive modeling through feature engineering and selection, model development and evaluation of multiple supervised ML algorithms, unsupervised optimizations, hyperparameter tuning, and provided visualizations to avoid black box models. 

The final model used is a XGBoost Random Forest model with R2 score of 0.95 and RMSE of 88.5m - an improvement of 75.0% over the baseline model. This indicates that the model has high accuracy of predicting future revenue from past sales data, enabling the company to be more responsive and confident in its revenue forecasting. 

Predictive modeling notebook: https://github.com/yunhonghe/realtime_dreamer/blob/main/notebooks/Model-Shipping-ProductPerformance-ReviewEmotion.ipynb

EDA notebook: https://github.com/yunhonghe/realtime_dreamer/blob/main/notebooks/Analytics-Shipping-Review.ipynb

<img width="500" height="150" alt="Predictive modeling - Feature Engineering and Selection" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Predictive-modeling-viz-featurecorrelation.jpg">
Feature Engineering and Selection

<img width="400" height="150" alt="Predictive modeling - Model Development and Evaluation" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Predictive-modeling-viz-modelevaluation.jpg">
Model Development and Evaluation

<img width="500" height="150" alt="Predictive modeling - Hyperparameter Tuning" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Predictive-modeling-final-model-optuna-optimizationhistory.jpg">
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
