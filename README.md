realtime_dreamer
==============================
<img width="1391" alt="Screen Shot 2565-08-19 at 23 56 15" src="https://user-images.githubusercontent.com/100912986/185674568-bfac3364-6555-4d8c-88c0-79101a23269e.png">

This Realtime Dreamer project is to use real Ecommerce data to perform NLP for Vietnamese language and machine learning tasks in order to settle a real-world business challenge of a company in Vietnam. We split our project into four tasks:



1.	Customer Reviews Type classification
 
Our  goal is to build a NLP model to predict the review classes using the Product_reviews_dataset.xlsx. The data collection process was completed by the customer service team of a company, who manually collected this data from the E-commerce platform. They recorded the review sentences (in Vietnamese) and the true rating score, along with labeling the type of the review manually

The pipeline script of this task created: reviewtype_script_.sh
 
 
Google Colab set up  and installation
To use PhoBERT model, we need GPU resources. Google Colab offers free GPU. Since we will be training a large neural network, we will need to take full advantage of this.

GPUs can be added by going to the menu and selecting: Edit -> Notebook Settings -> Add accelerator ( GPU )

Then run the cell below to confirm that the GPU has been received.
 
 <img width="395" alt="Screen Shot 2565-08-19 at 22 45 00" src="https://user-images.githubusercontent.com/100912986/185657079-c7f6c68e-d468-4913-bff8-f4d30baaa124.png">



For further analysis, the Vietnamese token was broken down from the larger original text. Many Vietnamese pre-trained NLP models are able to handle the task. For this task, we adopted a model called PhoBERT, which is a BERT base program (Bidirectional Encoder Representations from Transformers) released in late 2018. The objective of using this model  is to compare the performance of  PhoBERT  NLP to other traditional algorithms.

Generally, PhoBERT  NLP  is to  able to extract high-quality linguistic features from our review text data. or we apply this models for a specific task, such as classification, real-time recognition, and answer questions, among others. Pre-trained PhoBERT models are considered as the state-of-the-art language models for Vietnamese (Pho, i.e., "Phở", is a popular food in Vietnam).

To run the Phobert require installation  transformer and pytorch, so we have to install:

<img width="441" alt="Screen Shot 2565-08-19 at 22 46 25" src="https://user-images.githubusercontent.com/100912986/185759858-25d6f3c5-12b6-4a94-ae55-47b70de51c8a.png">

 
Data loading and preparation:

It will load, clean, and up-sample data to handle the two minority classes imbalanced. This process will take raw Reviews data  (stored in /data/raw/Git_mockup_review.xlsx. )


And output reviewType_pre_process.csv
 
 
Train_test_split the data 

We split data to Train and Test set , with test size =0.2 and then we split the Train data again to train_df and Val_df for model training and model loss validation, the process will take input of reviewType_pre_process.csv  and output reviewType_df_upload.csv ready to transmit to the dataloader function.
 
 
Train the Phobert model
 
reviewtype__train_test_val_split.py in src/data/ will  import PhoBERT's word separator (Tokenization) ,’vinai/phobert-base’ as below 
 <img width="1100" alt="Screen Shot 2565-08-19 at 23 15 18" src="https://user-images.githubusercontent.com/100912986/185662455-03f91b69-d32d-4d53-a724-07d8aaef50fb.png">


Then we select Phobert  Vietnamese NLP as Tokenizer


 <img width="553" alt="Screen Shot 2565-08-19 at 22 49 32" src="https://user-images.githubusercontent.com/100912986/185657737-cc1f708b-c317-48f7-a630-df0a2db08292.png">

 
which is used to convert the text into tokens corresponding to PhoBERT's lexicon. We load BertForSequenceClassification, which is a regular BERT model with a single linear layer added above for classification (Khanh, 2020). This process will be used to classify sentences. Once we provide the input data, the entire pre-trained BERT model and classifier will be trained with our specific task.


 <img width="464" alt="Screen Shot 2565-08-19 at 22 48 29" src="https://user-images.githubusercontent.com/100912986/185657647-227f228d-9160-4249-8cfe-94e8373da4f4.png">


We set the training loop by asking the model to compute gradience and putting the model in training mode, then unpack our input data. Then we delete the gradience in the previous iteration, Backpropagation, and update the weight using optimize.step() then, by each Epoch,we save the best model which has the lowest validation loss.

 
The result of each training epoch will be saved in the reviewType_pho_bert_eval_df.csv in the ‘data/interim’ directory,
 
Model tuning comparison
We create reviewtype_chart1_tuning.py to compare the result of the best Phobert model with different hyperparameter and data process tuning, the other models have been trained from Colab environments and uploaded thier the Evaluation_df into ‘data/external’ directory; we are focusing on the F1 macro score and,F1 Average score and validation lost of each trained Epoch,  the result is shown in the ‘report/’ directory.

![model_compar](https://user-images.githubusercontent.com/100912986/185656740-a483b745-edcc-4c71-bfaf-e94c755584c5.png)

 
The best model (number #5)  hyper parameter tuning recorded as below:
 
hidden_dropout_prob = 0.4

attention_probs_dropout_prob = 0.1
pre_trained_model = 'vinai/phobert-base'
model_type = pre_trained_model.split('/')[0]
batch_size = 8
epochs = 1
Ir = 1e-5
eps = 1e-8
 
 
Comparing Phobert with other algorithms
We also compare the result of all Phobert models with other traditional ML algorithms such as Random forest, SVM, and XGM classifier that are trained by using GridserchCv to tune hyperparameters. The best score from each parameter selected is imported to the ‘data/external’.

Again the best model is “Phobert- upsampling minority lass, which able to provide F1 macro score at 0.90)
![model_compare_traditional](https://user-images.githubusercontent.com/100912986/185656357-71bf8779-837d-4a76-b95a-3cb803bec2ab.png)


Model Testing 

we have run  several manual Vietnamese sentense testing ,for example , we input text to the model, and predict a class.

Input_text='Bàn ủi hơi nước cầm tay tiện lợi Tefal - DT6130E0, hàng chính hãng bảo hành 2 năm'

Predict review type :  Quality


In conclusion
We can conclude that we can use Phobert as a tokenizer and transform it to train the review data. Unlike the previous monolinguals and multilingual approaches, Phobert is superior in attaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of Dependency parsing, Named-entity recognition, Part-of-speech tagging, and Natural language inference. For this reason, it is the best algorithm to predict the reviews classification tasks because of its superiority compared to other algorithms. While the data imbalance was an issue due to moderate, we overcame it by over-sampling the minority. The outcome was optimal based on the elements of the task and no data preprocessing. The Phobert model requires parameter tuning, and from the results, we were able to increase hidden dropout to 0.4.







2.	Sentiment analysis for customer reviews

Yunhong He created reviews label dataset, trained and evaluated 3 Hugging Face Pre-trained BERT models including trituenhantaoio/bert-base-vietnamese-uncased, NlpHUST/vibert4news-base-cased, and bert-base-uncased, as well as Supervised Machine Learning algorithms, produced model evaluation visualizations. Setup team GitHub with folder structures. Created sentiment analysis Deep Learning and Machine Learning pipeline. Run /realtime_dreamer/sentiment_analysis.sh.

(1) Data used to train the model

Yunhong He used clear positive and negative keyword search and eye scan to select customer reviews for emotion labeling. There is imbalanced data in train reviews.xlsx. 8% of the labels are negative class while 87% of the labels are positive class.  As the task mainly focuses on finding the reviews with 4 or 5 ratings but have negative emotions, F1 macro score is important for model evaluation. Oversampling negative class by random.choices function can make the size of negative class as same as that of positive class.

Below are Model Evaluation Visualizations:

<img width="250" alt="Sentiment Analysis Imbalanced Classes" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20imbalanced%20classes%20before%20oversampling%20-%20visualization.png">

Graph 1: Imbalanced Emotion Classes

(2) Model Selection and Evaluation

Hugging Face pre-trained BERT models and Supervised machine learning algorithms are trained and evaluated in the sentiment analysis of the reviews.

<img width="1300" alt="Sentiment Analysis - Models" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20models.png">

Graph 2: Sentiment Analysis - Models


pre-trained BERT model are evaluated based on F1 scores (macro, micro and weighted), train and validation losses.

Model performance mainly focus on the F1 macro score due to the imbalanced label for the negative class. The best BERT model is defined as the pre-trained BERT model with the highest F1 macro score at the best epoch among 10 epochs. 

<img width="1300" alt="Screen Shot Sentiment Analysis Model Evaluation by Pre-Trained BERT Model"     src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20model%20evaluation%20by%20pre-trained%20bert%20model%20-%20visualization.png">

Graph 3: Sentiment Analysis Model Evaluation by Pre-trained BERT Model




<img width="1300" alt="Screen Shot Sentiment Analysis Model Evaluation by Metrics" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20model%20evaluation%20by%20metrics%20-%20visualization.png">

Graph 4: Sentiment Analysis Model Evaluation by Metrics





<img width="800" alt="Sentiment Analysis Model Evaluation by Classification Accuracy of Individual Classes" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20model%20evaluation%20by%20individual%20class%20prediction%20accuracy%20-%20visualization.png">

Graph 5: Sentiment Analysis Model Evaluation by Individual Class Prediction Accuracy


<img width="600" alt="Sentiment Analysis Supervised ML Classifier Model Evaluation" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20Emotions.png">



Graph 6: Sentiment Analysis Supervised ML Classifier Model Evaluation

Interactive graph: https://public.tableau.com/app/profile/agnes.he/viz/SentimentAnalysisforOnlineCustomerReviews/SentimentAnalysis-Emotions?publish=yes



<img width="600" alt="Sentiment Analysis - Emotions" src="https://public.tableau.com/app/profile/agnes.he/viz/SentimentAnalysisforOnlineCustomerReviews/SentimentAnalysis-Emotions?publish=yes">

Graph 7: Sentiment Analysis - Emotions


From below graph 8: Among the reviews with high rating of 4 and 5, 3% of them are actually negative, 35% are neutral.  This indicates that some of the online customers overstated the ratings to 4 and 5 while they actually have neutral or negative emotions.


<img width="600" alt="Sentiment Analysis - Emotions vs high ratings of 4 and 5" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20emotions%20vs%20high%20ratings%204%20and%205.png">

Graph 8: Sentiment Analysis - Emotions vs High Ratings

(3) Summary

Based on above Model Evaluation visualizations, it is found that trituenhantaoio pre-trained BERT model outperformed other pre-trained BERT models in terms of its stable and fast convergence of closing to zero for train loss, validation loss, and closing to 1 for F1 scores (macro, micro and weighted), and classification accuracy for positive, neutral and negative emotions across 10 epochs. Therefore model generated by trituenhantaoio at one of the 10 epochs with the highest F1 score macro is chosen as sentiment_analysis_best_bert_model.model.  The train data created by clear positive or negative keyword search and eye scan to select the reviews for emotion labeling can significantly improve the BERT model performance, and thus is used to train the BERT model in order to gain best performance of emotion classification for the company to better understand the needs of their customers in order to improve customer services and sales performance.


(4) Other information

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




3.	Recommendation system

Kensuke Suzuki used user id, product, and customer rating to train the Memory-Based Collaborative Filtering model which will then recommend items to the user. “Users who are similar to you also liked…” Conducted model evaluation.


4.	Machine Learning for Sales Prediction

Zheng Wei Lim used E-Commerce data to develop supervised ML algorithms through feature engineering, model evaluation of multiple supervised ML algorithms, unsupervised optimizations, hyperparameter tuning and visualizations. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    ├── README.md               <- The top-level README for developers using this project.
    ├── data
    │   ├── external            <- Data from third party sources.
    │       └── df__phobert_all.sav
    │       └── df__phobert_grouping.sav
    │       └── df_phobert_remove.sav
    │       └── df__phobert_all_upsamples.sav
    │       └── df_predict_all.sav
    │       └── df_predict_upsampling.sav
    │       └── df_predict_grouping.sav
    │       └── df_predict_remove.sav
    │       └── phobert1_eval_df_.csv
    │       └── phobert2_eval_df_.csv
    │       └── phobert3_eval_df_.csv
    │       └── phobert3_dropout_eval_df_.csv
    
    |   └── final               <- File with final data of the project
    |       └── reviewtype__accuracy_per_class_df.csv
    |       └── reviewtype_phobert_model.pt
    | 
    |
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
    |       https://drive.google.com/file/d/1ndzGpSsbzQ5mYRXkPMmzzg6bJmLLlD3q/view?usp=sharing
    │   └── sentiment_analysis_trituenhantaoio_train_data_provided_by_Yunhong He_NLP_Epoch10.model  
    |        <- link: https://drive.google.com/file/d/1ffLZd2jr5CGxGweuBq2bcM6lzIca66JB/view?usp=sharing
    │   └── predictive-final-model.sav <- final XGBoost Random Forest model for sales prediction    
    |
    ├── notebooks               <- Jupyter notebooks. A naming convention is a number (for ordering),
    │   |                          the creator's name, and a short `-` delimited description, e.g.
    │   |                          `1.0-jqp-initial-data-exploration.
    |   └── Sentiment_Analysis_Supervised_Machine_Learning_colab.ipynb   
    |       <- Complete Reviews label dataset is run in the notebook which can be run in Google Colab.
    |   └── Sentiment_Analysis_Supervised_Machine_Learning_Model_Evaluation_local.ipynb 
    |       <- Complete Reviews label dataset is run in this notebook.
    |   └── Sentiment_Analysis_BERT_Model_Evaluation.ipynb 
    |       <- Complete Reviews label dataset is run in this notebook.
    |   └── Sentiment_Analysis_BERT_Model_Evaluation.zip  
    |       <- Using zip file of the notebook to preserve the visualizations in the notebook.
    |   └── Model-Shipping-ProductPerformance-ReviewEmotion.ipynb  
    |       <- Predictive modeling on all merged data for sales forecasting
    |   └── Analytics-Shipping-Review.ipynb  <- EDA and visualization on shipping and review data
    │
    ├── references              <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    |   └── Sentiment Analysis.docx  <- Sentiment Analysis report
    |
    │   └── figures             <- Generated graphics and figures to be used in reporting
    |       └── sentiment analysis - model evaluation by individual class prediction accuracy - visualization.png
    |       └── sentiment analysis - model evaluation by metrics - visualization.png
    |       └── sentiment analysis - model evaluation by pre-trained bert model - visualization.png
    |       └── sentiment analysis - supervised ml model evaluation - visualization.png
    |       └── sentiment analysis - imbalanced classes before oversampling - visualization.png
    |       └── sentiment analysis - models
    |       └── reviewType_model_compare_traditional.png
    |       └── reviewType_model_compare.png
    |       └── confusion_phobert.png
    |        
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`
    │
    |── sentiment_analysis.sh   <- The bash file to run the sentiment_analysis pipeline.
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
    │   │
    │   └── visualization       <- Scripts to create exploratory and results-oriented visualizations
    │       └── reviewtype_chart1_tuning.py
    |       └── reviewtype_chart2_compare_traditional.py
    │
    └── tox.ini                 <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
