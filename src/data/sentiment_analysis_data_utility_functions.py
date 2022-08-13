import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import random
from sklearn.metrics import f1_score
import torch as torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import altair as alt
alt.renderers.enable('default')
import warnings
warnings.filterwarnings('ignore')
no_deprecation_warning=True

MAX_LEN = 768
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

label_dict = {'positive': 2, 'neutral': 1, 'negative': 0}
label_dict_inverse = {v: k for k, v in label_dict.items()}

model = build_Bert_model(pre_trained_model,attention_probs_dropout_prob,hidden_dropout_prob)
model.to(device)
epoch = best_epoch_F1_score_macro
best_model_path = f'emotion_{model_type}_NLP_{epoch}.model'
model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))

def import_review_data(df, sheet_name):
    df = pd.read_excel(
                        df,
                        engine='openpyxl',
                        sheet_name=sheet_name,
                        skiprows=0)
    return df


def oversample_minority_class(df):
    df = df.rename(columns={'Review Content': 'Vietnamese'})

    print('Count of individual class before oversamping negative class:')
    print(df['emotion'].value_counts())

    k_neg = len(df[df['emotion'] == 'positive']) \
            - len(df[df['emotion'] == 'negative'])

    new_index_neg = random.choices(df[df['emotion'] == 'negative']['index'].values,
                                   k=k_neg)
    df_add_neg = pd.DataFrame(new_index_neg, columns=['index'])
    df_add_neg_combined = pd.merge(df_add_neg, df,
                                   how='left',
                                   on=['index'])
    df = pd.concat([df, df_add_neg_combined], axis=0).reset_index()
    df.drop(['level_0'], axis=1, inplace=True)
    print('\nCount of individual class after oversamping negative class:')
    print(df['emotion'].value_counts())

    return df


def add_label_to_df(df):
    label_dict = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['label'] = df['emotion'].replace(label_dict)
    df = df[['index', 'Vietnamese', 'emotion', 'label']]
    return df


def data_split(df):
    df = add_label_to_df(df)
    X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                      df['label'].values,
                                                      test_size=0.20,
                                                      random_state=RANDOM_SEED,
                                                      stratify=df['label'].values)
    return X_train, X_val, y_train, y_val


def set_data_category_in_df(df):
    X_train, X_val, y_train, y_val = data_split(df)
    df['data_category'] = ['unset'] * df.shape[0]
    df.loc[X_train, 'data_category'] = 'train'
    df.loc[X_val, 'data_category'] = 'val'
    return df

def predict_text(input_text, tokenizer, model):
    inputs = tokenizer(input_text.lower(), return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return label_dict_inverse[predicted_class_id]


def predict_emotion_test():
    test_text_product = 'Sản phẩm rất tốt, và mạnh mẽ'  # Very good product, and powerful
    test_text_service = 'Tôi cần ai đó hỗ trợ tôi cách sử dụng'  # I need someone to help me how to use it
    test_text_logistic = 'giao hàng quá chậm'  # delivery is too slow
    print(" ")
    print("Predict emotion for 'Sản phẩm rất tốt, và mạnh mẽ': ", predict_text(test_text_product))
    print("Predict emotion for 'Tôi cần ai đó hỗ trợ tôi cách sử dụng': ", predict_text(test_text_service))
    print("Predict emotion for 'giao hàng quá chậm': ", predict_text(test_text_logistic))


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