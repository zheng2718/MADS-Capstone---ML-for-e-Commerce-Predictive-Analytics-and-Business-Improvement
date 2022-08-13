import pandas as pd
import sentiment_analysis_utility_functions as sauf

RANDOM_SEED = 42
col_name = 'emotion'
label_dict = {'positive': 2, 'neutral': 1, 'negative': 0}
label_dict_inverse = {v: k for k, v in label_dict.items()}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='review label dataset processed (csv)')
    parser.add_argument('output_file', help='review label dataset with data category (csv)')
    args = parser.parse_args()

    data_category_df = sauf.set_data_category_in_df(pd.read_csv(args.input_file))
    data_category_df.to_csv(args.output_file, index=False)
    print(data_category_df.data_category.value_counts())

