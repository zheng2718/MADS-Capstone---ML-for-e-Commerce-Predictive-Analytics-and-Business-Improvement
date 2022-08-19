realtime_dreamer
==============================
<img width="1391" alt="Screen Shot 2565-08-19 at 23 56 15" src="https://user-images.githubusercontent.com/100912986/185674568-bfac3364-6555-4d8c-88c0-79101a23269e.png">

This Realtime Dreamer project is to use real Ecommerce data to perform NLP for Vietnamese language and machine learning tasks in order to settle a real-world business challenge of a company in Vietnam. We split our project into four tasks:



##1.	Customer Reviews Type classification
 
This task is to build a NLP model to predict the review classes using the Reviews dataset. The data collection process was completed by the customer service team, who manually collected this data from the E-commerce platform. They recorded the review sentences (in Vietnamese) and the true rating score, labeling the type of the review manually

The pipeline script of this task created: reviewtype_script_.sh
 
 
Google Colab set up  and installation
To use PhoBERT model, we need GPU resources. Google Colab offers free GPU. Since we will be training a large neural network, we will need to take full advantage of this.

GPUs can be added by going to the menu and selecting: Edit -> Notebook Settings -> Add accelerator ( GPU )

Then run the cell below to confirm that the GPU has been received.
 
 <img width="395" alt="Screen Shot 2565-08-19 at 22 45 00" src="https://user-images.githubusercontent.com/100912986/185657079-c7f6c68e-d468-4913-bff8-f4d30baaa124.png">



For this task, we adopted a model called PHO_BERT, a BERT base program (Bidirectional Encoder Representations from Transformers) released in late 2018. We can use these models to extract high-quality linguistic features from our review text data, or we can refine these models for a specific task, such as classification, real-time recognition, and answer questions. Pre-trained PhoBERT models are the state-of-the-art language models for Vietnamese (Pho, i.e., "Phở", is a popular food in Vietnam). These keywords will determine the additional steps performed in this task.


<img width="441" alt="Screen Shot 2565-08-19 at 22 46 25" src="https://user-images.githubusercontent.com/100912986/185657495-4b1c93b2-90a3-4fc0-985c-152225ab1d64.png">
 
 
Data loading and preparation
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

Again the best model is “Phobert- upsampling minority lass, which able to provide F1 macro score at 0.88)
![model_compare_traditional](https://user-images.githubusercontent.com/100912986/185656357-71bf8779-837d-4a76-b95a-3cb803bec2ab.png)

In conclusion
We can conclude that we can use Phobert as a tokenizer and transform it to train the review data. Unlike the previous monolinguals and multilingual approaches, Phobert is superior in attaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of Dependency parsing, Named-entity recognition, Part-of-speech tagging, and Natural language inference. For this reason, it is the best algorithm to predict the reviews classification tasks because of its superiority compared to other algorithms. While the data imbalance was an issue due to moderate, we overcame it by over-sampling the minority. The outcome was optimal based on the elements of the task and no data preprocessing. The Phobert model requires parameter tuning, and from the results, we were able to increase hidden dropout to 0.4.



2.	Sentiment analysis for customer reviews

Yunhong He used the keyword search method to select positive and negative customer reviews and label them after an eye scan. And then used these selected reviews with labels to train trituenhantaoio/bert-base-vietnamese-uncased pre-trained BERT model which will label the customer reviews with emotions such as positive, neutral, and negative. Performed model evaluation and visualizations for 3 different pre-trained BERT models including trituenhantaoio/bert-base-vietnamese-uncased, NlpHUST/vibert4news-base-cased, and bert-base-uncased, as well as Supervised Machine Learning algorithms. Setup team GitHub with folder structures. Produced sentiment analysis Machine Learning pipeline. /realtime_dreamer/sentiment_analysis.sh file is used to run the sentiment analysis pipeline.

Model performance focus on the F1 macro score due to the imbalanced label for the negative class. The best BERT model is defined as the pre-trained BERT model with the highest F1 macro score at the best epoch among 10 epochs. It is found that trituenhantaoio pre-trained BERT model outperformed other pre-trained BERT models. Therefore model generated by trituenhantaoio at one of the 10 epochs with the highest F1 score macro is chosen as sentiment_analysis_best_bert_model.model. 

<img width="1391" alt="Screen Shot Sentiment Analysis Model Evaluation by Pre-Trained BERT Model" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20model%20evaluation%20by%20metrics%20-%20visualization.png">

<img width="1391" alt="Screen Shot Sentiment Analysis Model Evaluation by Metrics" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20model%20evaluation%20by%20metrics%20-%20visualization.png">

<img width="1391" alt="Screen Shot Sentiment Analysis Model Evaluation by Metrics" src="https://github.com/yunhonghe/realtime_dreamer/blob/main/reports/figures/sentiment%20analysis%20-%20model%20evaluation%20by%20metrics%20-%20visualization.png">


Below are the links to the sentiment analysis BERT models generated.

Links:
(1) The link to /models/sentiment_analysis_best_bert_model.model is https://drive.google.com/file/d/1ndzGpSsbzQ5mYRXkPMmzzg6bJmLLlD3q/view?usp=sharing

(2) The link to /models/sentiment_analysis_trituenhantaoio_train_data_provided_by_Yunhong He_NLP_Epoch10.model is  https://drive.google.com/file/d/1ffLZd2jr5CGxGweuBq2bcM6lzIca66JB/view?usp=sharing

Sample files:

(1) Customer review label dataset at C:\Users\heyun\Capstone\realtime_dreamer\data\processed\sentiment_analysis_reviews_label.xlsx is the small sample of the file '/content/drive/MyDrive/Realtime Dreamer/train reviews.xlsx' used in notebooks\Sentiment_Analysis_BERT_Model_Evaluation.ipynb. Column "emotion" is labeled by Yunhong He using the keyword search method to select positive and negative customer reviews after an eye scan.

(2) Customer reviews dataset at C:\Users\heyun\Capstone\realtime_dreamer\data\processed\Git_mockup_reviews_processed.xlsx is the small sample of the customer review file 'drive/MyDrive/Realtime Dreamer/Tefal Lazada Product Reviews in TTL202207_Updated_Good_Bad.xlsx' used in notebooks\Sentiment_Analysis_BERT_Model_Evaluation.ipynb. Column "Comment classified Type 1" is labeled by the Vietnamese team.

BERT model evaluation files:

Below evaluation files are generated in Sentiment_Analysis_BERT_Model_Evaluation.ipynb and Sentiment_Analysis_BERT_Model_Evaluation.zip, and are used to produce BERT model evaluation visualizations:

model_info.csv, sentiment_analysis_...accuracy_per_class_df.csv, sentiment_analysis_...eval_df.csv, sentiment_analysis_...eval_VnEmoLex_validated_df.csv,  sentiment_analysis_...accuracy_per_class_VnEmoLex_validated_df.csv, sentiment_analysis_...eval_before_oversample_df.csv, and sentiment_analysis_...accuracy_per_class_before_oversample_df.csv.


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
