#:/bin/sh
echo ""
echo "This is sentiment analysis pipeline"
echo ""
echo "Step 1: Import and prepare reviews label data..."
echo ""
python src/data/sentiment_analysis_prepare_review_label.py data/processed/sentiment_analysis_reviews_label.xlsx data/processed/sentiment_analysis_reviews_label_processed.csv
wait

echo ""
echo ""
echo "Step 2: Categorize train and validation inside reviews label data..."
echo ""
python src/data/sentiment_analysis_split_train_val_data.py data/processed/sentiment_analysis_reviews_label_processed.csv data/processed/sentiment_analysis_reviews_label_split.csv
wait

echo ""
echo ""
echo "Step 3: Train Bert Model..."
echo ""
python src/models/sentiment_analysis_train_bert_model.py data/processed/sentiment_analysis_reviews_label_split.csv
wait

echo ""
echo ""
echo "Step 4: Predict emotion in customer reviews..."
python src/data/sentiment_analysis_predict_emotion.py  data/raw/Git_mockup_reviews.xlsx data/interim/sentiment_analysis_reivew_emotion_predition.xlsx
wait




