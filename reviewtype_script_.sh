#:/bin/sh

echo "This is Reviews type prediction (task1 ) pipeline"

echo ""
echo "Step 1: Import and prepare reviews label data..."
echo ""

python src/data/reviewtype__prepare_review_label.py data/processed/Reviews.xlsx data/processed/reviewType_pre_process.csv
echo ""


echo "Split data for training "
python src/data/reviewtype__train_test_val_split.py data/processed/reviewType_pre_process.csv data/processed/reviewType_df_upload.csv

echo "---I will take around 5-10minute per epoch, please wait "
echo "start training .may take few minutes "
python src/models/reviewtype_train_phobert.py data/processed/reviewType_df_upload.csv
echo "your model is ready "

echo "validating the model"
python src/models/reviewtype_validate_phobert_model.py data/processed/reviewType_df_upload.csv

echo "testing model"
echo " You can input a Vietnamese sentences here"
python src/models/reviewtype_text_test_prediction.py 'Sản phẩm rất tốt, và mạnh mẽ' 

echo "compare with other models"
python src/visualization/reviewtype_chart1_tuning.py data/interim/reviewType_pho_bert_eval_df.csv


echo "compare with other algorithm"

python src/visualization/git_pipeline_capstone/reviewtype_chart2_compare_traditional.py

echo "End of pipeline"
