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


# %%
def encode_data_and_prepare_dataset(df, tokenizer):
    df = set_data_category_in_df(df)
    encoded_data_train = tokenizer.batch_encode_plus(df[df.data_category == 'train'].Vietnamese.values,
                                                     add_special_tokens=True,
                                                     return_attention_mask=True,
                                                     padding=True,
                                                     return_tensors='pt')

    encoded_data_val = tokenizer.batch_encode_plus(df[df.data_category == 'val'].Vietnamese.values,
                                                   add_special_tokens=True,
                                                   return_attention_mask=True,
                                                   padding=True,
                                                   return_tensors='pt')

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(df[df.data_category == 'train']['label'].values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(df[df.data_category == 'val']['label'].values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
    return dataset_train, dataset_val


# %%
def build_Bert_model(pre_trained_model, attention_probs_dropout_prob, hidden_dropout_prob):
    label_dict = {'positive': 2, 'neutral': 1, 'negative': 0}
    model = BertForSequenceClassification.from_pretrained(pre_trained_model,
                                                          num_labels=len(label_dict),
                                                          output_attentions=False,
                                                          output_hidden_states=False,
                                                          hidden_dropout_prob=hidden_dropout_prob,
                                                          attention_probs_dropout_prob=attention_probs_dropout_prob
                                                          )
    return model


def build_dataloader(df, batch_size, tokenizer):
    dataset_train, dataset_val = encode_data_and_prepare_dataset(df, tokenizer)
    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)
    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)
    return dataloader_train, dataloader_validation


def setup_optimizer(model, Ir, eps):
    optimizer = AdamW(model.parameters(),
                      lr=Ir,
                      eps=eps)
    return optimizer


def setup_scheduler(dataloader_train, optimizer, epochs):
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)
    return scheduler


def evaluate(model, dataloader_validation):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_validation:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_validation)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def train_model(model, pre_trained_model, model_type, Ir, eps, attention_probs_dropout_prob, hidden_dropout_prob, epochs,
                batch_size, dataloader_train, dataloader_validation, train_data_provider):
    epoch_list = []
    train_loss = []
    validation_loss = []
    F1_score_weighted = []
    F1_score_macro = []
    F1_score_micro = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    optimizer = setup_optimizer(model, Ir, eps)

    scheduler = setup_scheduler(dataloader_train, optimizer, epochs)

    for epoch in tqdm(range(1, epochs + 1)):

        epoch_list.append(epoch)
        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

        for batch in progress_bar:
            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2],
                      }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

        torch.save(model.state_dict(), f'c:\\Users\\heyun\\Capstone\\realtime_dreamer\\data\\processed\\sentiment_analysis_emotion_{model_type}_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        train_loss.append(loss_train_avg)

        val_loss, predictions, true_vals = evaluate(model, dataloader_validation)
        val_f1_weighted = f1_score_func(predictions, true_vals, 'weighted')
        val_f1_macro = f1_score_func(predictions, true_vals, 'macro')
        val_f1_micro = f1_score_func(predictions, true_vals, 'micro')
        validation_loss.append(val_loss)
        F1_score_weighted.append(val_f1_weighted)
        F1_score_macro.append(val_f1_macro)
        F1_score_micro.append(val_f1_micro)

        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1_weighted}')
        tqdm.write(f'F1 Score (macro): {val_f1_macro}')
        tqdm.write(f'F1 Score (micro): {val_f1_micro}')

        eval_df = pd.DataFrame()
        eval_df['emotion'] = ['positive,negative,neutral'] * len(epoch_list)
        eval_df['epoch'] = epoch_list
        eval_df['train_loss'] = train_loss
        eval_df['val_loss'] = validation_loss
        eval_df['F1_score_weighted'] = F1_score_weighted
        eval_df['F1_score_macro'] = F1_score_macro
        eval_df['F1_score_micro'] = F1_score_micro
        eval_df['batch_size'] = batch_size
        eval_df['Ir'] = Ir
        eval_df['eps'] = eps
        eval_df['pre_trained_model'] = pre_trained_model
        eval_df['hidden_dropout_prob'] = hidden_dropout_prob
        eval_df['attention_probs_dropout_prob'] = attention_probs_dropout_prob
        eval_df['note'] = 'Added and splited Self-judged Review emotions into train and valiation by 8:2'
        eval_df['train_data_creator'] = train_data_provider

    return eval_df, model


def f1_score_func(preds, labels, average):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average=average)


def accuracy_per_class(preds, labels, model_type, epochs, train_data_provider):
    class_list = []
    score_list = []
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    print("Prediction accuracy for individual class:")
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        acc = len(y_preds[y_preds == label]) / len(y_true)
        class_list.append(label_dict_inverse[label])
        score_list.append(len(y_preds[y_preds == label]) / len(y_true))
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}, ', \
              '{:.3f}'.format(acc), '\n')
    df = pd.DataFrame(class_list, columns=['class'])
    df['score'] = score_list
    df['pre_trained_model'] = model_type
    df['epoch'] = epochs
    df['train_data_creator'] = train_data_provider
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


def prepare_model_df(eval_df_path_list, accuracy_per_class_path_list, model_path_list, best_epoch_f1_score_macro_list):
    model_df = pd.DataFrame()
    model_df['eval_df_path'] = eval_df_path_list
    model_df['accuracy_per_class_path'] = accuracy_per_class_path_list
    model_df['model_path'] = model_path_list
    model_df['best_epoch_F1_score_macro'] = best_epoch_f1_score_macro_list
    model_df.to_csv(f'c:\\Users\\heyun\\Capstone\\realtime_dreamer\\data\\interim\\sentiment_analysis_model_info.csv',
                    index=False)
    return model_df


def draw_F1_scores(df, domain, width, height, first_line_title, second_line_title, epochs):
    if epochs == 1:
        base = alt.Chart(df).mark_point()
    else:
        base = alt.Chart(df).mark_line()

    graph = base.encode(
        x='epoch',
        y=alt.Y("value:Q", title="", scale=alt.Scale(domain=domain)),
        color=alt.Color('line type:N',
                        legend=alt.Legend(
                            title="Metrics",
                            orient='right',
                            titleFontSize=11,
                            titleColor='black',
                            labelFontSize=10.5,
                            labelColor='black',
                            direction='vertical')),
        tooltip=['emotion', 'epoch', 'line type', 'value']
    ).interactive(
    ).properties(
        width=width,
        height=height,
        title={"text": [first_line_title, second_line_title], "color": "black"})
    return graph


def draw_prediction_accuracy(df, domain, width, height, first_line_title, second_line_title):
    base = alt.Chart(df).mark_bar().encode(
        x=alt.X('class:N', axis=alt.Axis(labelAngle=360))
    ).properties(
        width=width,
        height=height,
        title={"text": [first_line_title, second_line_title], "color": "black"})

    graph = base.mark_bar(size=20).encode(
        y=alt.Y("score:Q", title="", scale=alt.Scale(domain=domain)),
        color=alt.Color('class:N', legend=None),
        tooltip=['pre_trained_model', 'class', 'score', 'epoch', 'train_data_creator']
    ).interactive()
    return graph


def prepare_data_for_metrics_graphs(df, best_epoch_F1_score_macro, best_model=False):
    df_f1_scores_id_vars = ['emotion', 'epoch', 'batch_size', 'Ir', 'eps', 'pre_trained_model', 'hidden_dropout_prob',
                            'attention_probs_dropout_prob', 'note', 'train_data_creator']
    df_f1_scores_value_vars = ['train_loss', 'val_loss', 'F1_score_weighted', 'F1_score_macro', 'F1_score_micro']


    df_f1_scores_var_name = ['line type']
    if best_model:
        df_f1_scores = df[df['epoch'] <= best_epoch_F1_score_macro]
    else:
        df_f1_scores = df.copy()
    df_f1_scores_long = pd.melt(df_f1_scores,
                                id_vars=df_f1_scores_id_vars,
                                value_vars=df_f1_scores_value_vars,
                                var_name=df_f1_scores_var_name)
    return df_f1_scores_long


def prepare_metrics_graphs(df_f1_scores_long, df_class_accuracy, domain, width, height, epochs):
    F1_scores_graph = draw_F1_scores(df_f1_scores_long,
                                     domain,
                                     width,
                                     height,
                                     "F1 scores (Macro, Micro, Weighted)",
                                     "Train loss, Validation loss",
                                     epochs)

    prediction_accuracy_graph = draw_prediction_accuracy(df_class_accuracy,
                                                         domain,
                                                         width,
                                                         height,
                                                         "Accuracy Per Class",
                                                         "Pre-Trained Model: " \
                                                         + df_class_accuracy['pre_trained_model'][1])
    return F1_scores_graph, prediction_accuracy_graph