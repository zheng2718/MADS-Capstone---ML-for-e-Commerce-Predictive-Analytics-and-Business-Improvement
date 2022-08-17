import sentiment_analysis_utility_functions as sauf
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import random
import torch
from transformers import BertTokenizer, logging
import altair as alt
alt.renderers.enable('default')
import warnings
warnings.filterwarnings('ignore')
no_deprecation_warning=True
logging.set_verbosity_error()

MAX_LEN = 768
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_dropout_prob = 0.1
attention_probs_dropout_prob = 0.1
pre_trained_model = 'trituenhantaoio/bert-base-vietnamese-uncased'
model_type = pre_trained_model.split('/')[0]
batch_size = 2
epochs = 10
Ir = 1e-5
eps = 1e-8
train_data_provider = 'Yunhong He'

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='sentiment analysis reviews label data with train and validation split(csv)')
    args = parser.parse_args()

    reviews_label_df = pd.read_csv(args.input_file)
    print(reviews_label_df['label'].value_counts())

    tokenizer = BertTokenizer.from_pretrained(pre_trained_model, do_lower_case=True)
    dataloader_train, dataloader_validation = sauf.build_dataloader(reviews_label_df, batch_size, tokenizer)

    model = sauf.build_Bert_model(pre_trained_model, attention_probs_dropout_prob, hidden_dropout_prob)
    

    eval_df, model = sauf.train_model(model, pre_trained_model, model_type, Ir, eps, attention_probs_dropout_prob, hidden_dropout_prob,
                                      epochs, batch_size, dataloader_train, dataloader_validation, train_data_provider)
    eval_df_path = f'c:\\Users\\heyun\\Capstone\\realtime_dreamer\\data\\interim\\sentiment_analysis_{model_type}_Epoch{epochs}_train_data_from_{train_data_provider}_eval_df.csv'
    eval_df.to_csv(eval_df_path , index=False)

    model_path = f'c:\\Users\\heyun\\Capstone\\realtime_dreamer\\models\\sentiment_analysis_{model_type}_train_data_from_{train_data_provider}_Epoch{epochs}.model'
    torch.save(model.state_dict(), model_path)

    val_df_path = f'c:\\Users\\heyun\\Capstone\\realtime_dreamer\\data\\interim\\sentiment_analysis_{model_type}_Epoch{epochs}_train_data_from_{train_data_provider}_eval_df.csv'
    _, predictions, true_vals = sauf.evaluate(model, dataloader_validation)
    accuracy_per_class_df = sauf.accuracy_per_class(predictions, true_vals, model_type, epochs, train_data_provider)
    accuracy_per_class_df_path = f'c:\\Users\\heyun\\Capstone\\realtime_dreamer\\data\\interim\\sentiment_analysis_{model_type}_train_data_from_{train_data_provider}_Epoch{epochs}_accuracy_per_class_df.csv'
    accuracy_per_class_df.to_csv(accuracy_per_class_df_path, index=False)

    best_epoch_F1_score_macro = eval_df[eval_df['F1_score_macro'] == max(eval_df['F1_score_macro'])]['epoch'].values[0]
    print('best epoch for the best F1 score (macro): ', best_epoch_F1_score_macro)

    best_model_path = f'c:\\Users\\heyun\\Capstone\\realtime_dreamer\\models\\sentiment_analysis_emotion_{model_type}_{best_epoch_F1_score_macro}.model'
    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))

    best_model_path_interim = f'c:\\Users\\heyun\\Capstone\\realtime_dreamer\\models\\sentiment_analysis_best_bert_model.model'
    torch.save(model.state_dict(), best_model_path_interim)
