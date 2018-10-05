import sys
import pandas as pd
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    """

    :param messages_filepath: Path to the data file, which contains the messages data.
    :param categories_filepath: Path to the data file, which contains the categories data.
    :return: df: Dataframe of the merged data files.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='outer')
    return df

def clean_data(df):
    """
    Clean the data set.
    All categories are in one column in the input df, separated by semicolons.
    These categories are extracted and each of them becomes a column.
    The last character of category name holds the value (zero or one).
    The zeros and ones will be the values in the frame.

    The new 'categories' data frame replaces the 'categories' column in the input data frame.

    Duplicated rows are dropped based on their 'id' keys.

    :param df: data frame, containing all raw data.
    :return: Cleaned data frame.
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(subset='id', inplace=True)
    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database.

    :param df: The data frame to save.
    :param database_filename: Destination file name.
    """

    engine = sqlalchemy.create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, if_exists='replace', index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
