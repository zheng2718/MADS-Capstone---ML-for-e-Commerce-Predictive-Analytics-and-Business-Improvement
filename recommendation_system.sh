echo ""
echo "This is the recommendation system pipeline"
echo ""
echo "Step 1: Predict recommended items for User 52354"
echo ""
python src/data/recommendation_system.py data/processed/reviews_with_user_id.csv 52354 data/processed/recommendation_for_user_52354.csv
wait
