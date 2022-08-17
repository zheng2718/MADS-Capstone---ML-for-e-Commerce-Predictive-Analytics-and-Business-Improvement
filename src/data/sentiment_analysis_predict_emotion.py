import sentiment_analysis_utility_functions as sauf
import pandas as pd
import torch
from transformers import BertTokenizer, logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_dropout_prob = 0.1
attention_probs_dropout_prob = 0.1
pre_trained_model = 'trituenhantaoio/bert-base-vietnamese-uncased'
tokenizer = BertTokenizer.from_pretrained(pre_trained_model, do_lower_case=True)
label_dict = {'positive': 2, 'neutral': 1, 'negative': 0}
label_dict_inverse = {v: k for k, v in label_dict.items()}
model = sauf.build_Bert_model(pre_trained_model, attention_probs_dropout_prob, hidden_dropout_prob)
best_model_path_interim = f'c:\\Users\\heyun\\Capstone\\realtime-dreamer\\data\\interim\\sentiment_analysis_trituenhantaoio_train_data_provided_by_Yunhong He_NLP_Epoch10.model'
realtime_best_model_path_interim = f'c:\\Users\\heyun\\Capstone\\realtime_dreamer\\data\\interim\\sentiment_analysis_best_bert_model.model'
model.load_state_dict(torch.load(best_model_path_interim, map_location=torch.device('cpu')))

def predict_text(input_text):
    inputs = tokenizer(input_text.lower(), return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return label_dict_inverse[predicted_class_id]

def prepare_review_data(review_df):
    review_df = review_df.reset_index()
    review_df_1 = review_df[['index', 'Review Content', 'Rating']].dropna(how='any')
    print("raw review dataset after dropping null value:", review_df_1.shape)
    return review_df, review_df_1

def generate_predition_data(df):
    review_df, review_df1 = prepare_review_data(df)
    review_df1['emotion'] = review_df1['Review Content'].apply(predict_text)
    review_emotion_prediction = pd.merge(review_df, review_df1[['index', 'emotion']], how='left', on='index')
    return review_emotion_prediction

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='review sample dataset (excel)')
    parser.add_argument('output_file', help='reviews emotion prediction(excel)')
    args = parser.parse_args()

    import_data = sauf.import_review_data(args.input_file, 'Database_LZD')
    review_emotion_prediction = generate_predition_data(import_data)
    review_emotion_prediction.to_excel(args.output_file, index=False)
