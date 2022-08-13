import sentiment_analysis_utility_functions as sauf

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='review label dataset (excel)')
    parser.add_argument('output_file', help='processed review label dataset (csv)')
    args = parser.parse_args()

    import_data = sauf.import_review_data(args.input_file, 'train')
    output_data = sauf.oversample_minority_class(import_data)
    output_data.to_csv(args.output_file, index=False)
