<h1>realtime_dreamer</h1>
==============================
<img width="1391" alt="Screen Shot 2565-08-19 at 23 56 15" src="https://user-images.githubusercontent.com/100912986/185674568-bfac3364-6555-4d8c-88c0-79101a23269e.png">

This Realtime Dreamer project is to use real Ecommerce data to perform NLP for Vietnamese language and machine learning tasks in order to settle a real-world business challenge of a company in Vietnam. We split our project into four tasks:

For the full report,please visit our blog post :https://madsrealtimedreamer.wordpress.com/

<h2>1.	Customer Reviews Type classification</h2>
 
Our  goal is to build a NLP model to predict the review classes using the customer reviews (sample data: Git_mockup_reviews.xlsx). The data collection process was completed by the customer service team of the company, who manually collected this data from the E-commerce platform. They recorded the review sentences (in Vietnamese) and the true rating score, along with labeling the type of the review manually

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
         output_hidden_states = False
        )
      model = model.to(device)


We set the training loop by asking the model to compute gradience and putting the model in training mode, then unpack our input data. Then we delete the gradience in the previous iteration, Backpropagation, and update the weight using optimize.step() then, by each Epoch,we save the best model which has the lowest validation loss.

 
The result of each training epoch will be saved in the reviewType_pho_bert_eval_df.csv in the ‘data/interim’ directory,
 
<strong><br/>Model tuning comparison</strong><br/>
We create reviewtype_chart1_tuning.py to compare the result of the best Phobert model with different hyperparameter and data process tuning, the other models have been trained from Colab environments and uploaded thier the Evaluation_df into ‘data/external’ directory; we are focusing on the F1 macro score and,F1 Average score and validation lost of each trained Epoch,  the result is shown in the ‘report/’ directory.

![reviewType_model_compare (2)](https://user-images.githubusercontent.com/100912986/185786647-6143751e-93fc-4076-bbcd-a79fd6d4c555.png)


 
The best model (number #5)  hyper parameter tuning recorded as below:
 
hidden_dropout_prob = 0.4

attention_probs_dropout_prob = 0.1
pre_trained_model = 'vinai/phobert-base'
model_type = pre_trained_model.split('/')[0]
batch_size = 8
epochs = 2
Ir = 1e-5
eps = 1e-8
 
 
<br/><strong>Comparing Phobert with other algorithms</strong><br/>
We also compare the result of all Phobert models with other traditional ML algorithms such as Random forest, SVM, and XGM classifier that are trained by using GridserchCv to tune hyperparameters. The best score from each parameter selected is imported to the ‘data/external’.

Again the best model is “Phobert- upsampling minority lass, which able to provide F1 macro score at 0.90)

![reviewType_model_compare_traditional (1)](https://user-images.githubusercontent.com/100912986/185786652-38bb2353-2fe8-4791-903b-f47697751be3.png)

<strong><br/>Model Testing </strong><br/>

we have run  several manual Vietnamese sentense testing ,for example , we input text to the model, and predict a class.

Input_text='Bàn ủi hơi nước cầm tay tiện lợi Tefal - DT6130E0, hàng chính hãng bảo hành 2 năm'

Predict review type :  Quality


<strong>In conclusion</strong><br/>
We can conclude that we can use Phobert as a tokenizer and transform it to train the review data. Unlike the previous monolinguals and multilingual approaches, Phobert is superior in attaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of Dependency parsing, Named-entity recognition, Part-of-speech tagging, and Natural language inference. For this reason, it is the best algorithm to predict the reviews classification tasks because of its superiority compared to other algorithms. While the data imbalance was an issue due to moderate, we overcame it by over-sampling the minority. The outcome was optimal based on the elements of the task and no data preprocessing. The Phobert model requires parameter tuning, and from the results, we were able to increase hidden dropout to 0.4.


<strong>  </strong><br/>

<strong>  </strong><br/>


<h2>2.	Sentiment Analysis For Customer Reviews</h2>

Yunhong He created reviews label dataset, trained and evaluated 3 Hugging Face Pre-trained BERT models including trituenhantaoio/bert-base-vietnamese-uncased, NlpHUST/vibert4news-base-cased, and bert-base-uncased, as well as Supervised Machine Learning algorithms, produced model evaluation visualizations. Setup team GitHub with folder structures. Created sentiment analysis Deep Learning and Machine Learning pipeline. Run /realtime_dreamer/sentiment_analysis.sh.

<img width="1300" alt="Sentiment Analysis - Models" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Sentiment%20analysis%20-%20process.png">

Graph 1: Sentiment Analysis Process

<strong>  </strong><br/>

<strong>(1) Data used to train the model</strong><br/>

Yunhong He used clear positive and negative keyword search and eye scan to select customer reviews for emotion labeling. 

<img width="700" alt="Sentiment Analysis Data Preprocessing" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20Data%20preprocess.png">

Graph 2: Sentiment Analysis Data Pre-processing

There is imbalanced data in train reviews.xlsx. 8% of the labels are negative class while 87% of the labels are positive class.  As the task mainly focuses on finding the reviews with 4 or 5 ratings but have negative emotions, F1 macro score is important for model evaluation. Oversampling negative class by random.choices function can make the size of negative class as same as that of positive class.


<img width="250" alt="Sentiment Analysis Imbalanced Classes" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20imbalanced%20classes%20before%20oversampling%20-%20visualization.png">

Graph 3: Imbalanced Emotion Classes

<strong>  </strong><br/>

<strong>(2) Model Selection and Evaluation</strong><br/>

Hugging Face pre-trained BERT models and Supervised machine learning algorithms are trained and evaluated in the sentiment analysis of the reviews.

<img width="900" alt="Sentiment Analysis - Models" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20models.png">

Graph 4: Sentiment Analysis - Models

<strong>  </strong><br/>

pre-trained BERT model are evaluated based on F1 scores (macro, micro and weighted), train and validation losses.

<img width="1000" alt="Sentiment Analysis - Models" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20BERT%20Model%20training%20-%20Fine%20Tuning%20and%20Evaluation.png">

Graph 5: Sentiment Analysis BERT Model Training, Fine Tuning and Evaluation

Model performance mainly focus on the F1 macro score due to the imbalanced label for the negative class. The best BERT model is defined as the pre-trained BERT model with the highest F1 macro score at the best epoch among 10 epochs. 


From below Model Evaluation Visualizations, we can see that trituenhantaoio pre-trained BERT model outperformed other pre-trained BERT models in terms of its stable and fast convergence of closing to zero for train loss, validation loss, and closing to 1 for F1 scores (macro, micro and weighted), and classification accuracy for positive, neutral and negative emotions across 10 epochs. Moreover, the train data created by clear positive or negative keyword search and eye scan to select the reviews for emotion labeling and oversamping the negative class can significantly improve the BERT model performance.


<img width="1300" alt="Screen Shot Sentiment Analysis Model Evaluation by Pre-Trained BERT Model"     src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20model%20evaluation%20by%20pre-trained%20bert%20model%20-%20visualization.png">

Graph 6: Sentiment Analysis Model Evaluation by Pre-trained BERT Model


<img width="1300" alt="Screen Shot Sentiment Analysis Model Evaluation by Metrics" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20model%20evaluation%20by%20metrics%20-%20visualization.png">

Graph 7: Sentiment Analysis Model Evaluation by Metrics


<strong>  </strong><br/>


<img width="800" alt="Sentiment Analysis Model Evaluation by Classification Accuracy of Individual Classes" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20model%20evaluation%20by%20individual%20class%20prediction%20accuracy%20-%20visualization.png">

Graph 8: Sentiment Analysis Model Evaluation by Individual Class Prediction Accuracy

<strong>  </strong><br/>


Supervised machine learning algorithms are evaluated based on F1 scores (macro, micro, and weighted)

<img width="1000" alt="Sentiment Analysis Supervised ML Classifier Model Evaluation" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20Supervised%20Machine%20Learning%20Process.png">

Graph 9: Sentiment Analysis - Supervised Machine Learning Process

<strong>  </strong><br/>

<img width="600" alt="Sentiment Analysis Supervised ML Classifier Model Evaluation" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20supervised%20ml%20model%20evaluation%20-%20visualization.png">

Graph 10: Sentiment Analysis Supervised ML Classifier Model Evaluation

<strong>  </strong><br/>

From below Graph 11: Sentiment Analysis - Emotions vs Ratings, we can see that the customer emotion in 39% of total reviews are actually neutral but gave the highest average ratings of 5. 7% of the toal reviews are negative but have average rating of 3.6 above median level.

<img width="600" alt="Sentiment Analysis - Emotions" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20Emotions%20vs%20rating.png">

Graph 11: Sentiment Analysis - Emotions vs Ratings

Above interactive graph is in Tab "Emotions Dashboard" at below link: https://public.tableau.com/app/profile/agnes.he/viz/SentimentAnalysisforOnlineCustomerReviews/SentimentAnalysisforOnlineCustomerReviewsDashBoard?publish=yes

<strong>  </strong><br/>

As shown in graph 12: Sentiment Analysis - Emotions vs High Ratings, among the reviews with high rating of 4 and 5, 4% of them are actually negative, 41% are neutral.  This indicates that 45% of the online customers overstated the ratings to 4 and 5 while they actually have neutral or negative emotions.

<img width="600" alt="Sentiment Analysis - Emotions vs high ratings of 4 and 5" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20emotions%20vs%20higher%20ratings%204%20and%205.png">

Graph 12: Sentiment Analysis - Emotions vs High Ratings

Above interactive graph is in Tab "Emotions Dashboard" at below link: https://public.tableau.com/app/profile/agnes.he/viz/SentimentAnalysisforOnlineCustomerReviews/SentimentAnalysisforOnlineCustomerReviewsDashBoard?publish=yes



<strong>  </strong><br/>

As shown in the below Graph 13 Sentiment Analysis Dashboard, positive reviews mainly come from Product (24%) and Quality (21%). Customers are quite happy with Products with 95% of positive emotion and average rating of 4.9.  50% of Sales reviews, 18% of Logistic reviews, 13% of Service reviews are negative, Their average of rating for negative emotion was 3.6,  2.9 and 3.5,  which indicates that customers showed their dissatisfaction mainly in Sales, and then Logistic  and Service. Service reviews has lowest average rating in negative class. Additionally, 60% of Quality related reviews indicates neutral emotion.


<img width="1300" alt="Sentiment Analysis - Emotions Dashboard" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20emotions%20vs%20content%20class%20dashboard.png">

Graph 13: Sentiment Analysis Dashboard

Above interactive sentiment analysis visualization dashboard is in Tab "Sentiment Analysis Dashboard" at below link: https://public.tableau.com/app/profile/agnes.he/viz/SentimentAnalysisforOnlineCustomerReviews/SentimentAnalysisforOnlineCustomerReviewsDashBoard?publish=yes

<strong>  </strong><br/>

<strong>(3) Summary</strong><br/>

Based on above Model Evaluation visualizations, it is found that trituenhantaoio pre-trained BERT model outperformed other pre-trained BERT models in terms of its stable and fast convergence of closing to zero for train loss, validation loss, and closing to 1 for F1 scores (macro, micro and weighted), and classification accuracy for positive, neutral and negative emotions across 10 epochs. Therefore model generated by trituenhantaoio at one of the 10 epochs with the highest F1 score macro is chosen as sentiment_analysis_best_bert_model.model.  The train data created by clear positive or negative keyword search and eye scan to select the reviews for emotion labeling and oversamping the negative class can significantly improve the BERT model performance, and thus this method is used to train the BERT model in order to gain best performance of emotion classification for the company to better understand the needs of their customers in order to improve their logistic, customer services and sales performance.

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

Kensuke Suzuki used user id, product, and customer rating to train the Memory-Based Collaborative Filtering model which will then recommend items to the user. “Users who are similar to you also liked…”.

To run the SVD algorithm for the recommender system, you will be required to install a python library called surprise: 

```pip install surprise```

After installing Surprise, run 
```recommendation_system.sh```
and the final output recommendation_for_user_52354.csv will be in the data/final folder. 


<h2>4.	Machine Learning for Sales Prediction</h2>

Zheng Wei Lim used data of E-Commerce sales, customer, and product data to develop supervised ML algorithms through feature engineering and selection, model evaluation of multiple supervised ML algorithms, unsupervised optimizations, hyperparameter tuning and visualizations. 
The final model used is a XGBoost Random Forest model with R2 score of 0.95 and RMSE of 82.7m - an improvement of 77.8% over the baseline model. This indicates that the model has high accuracy of predicting future revenue from past sales data, enabling the company to be more responsive and confident in its revenue forecasting. 

<img width="464" alt="Predictive modeling - feature engineering and selection" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/Predictive-modeling-viz-features-correlation.jpg">



Project Organization
------------

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
    |       └── sentiment analysis - models
    |       └── reviewType_model_compare_traditional.png
    |       └── reviewType_model_compare.png
    |       └── confusion_phobert.png
    |        
    │
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
