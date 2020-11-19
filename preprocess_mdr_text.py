#######################################################################
#  Title:   preprocess_mdr_text.py
#
#  Purpose: Perform simple text preprocessing on MDR data
#
#  Author:  Aaron Boussina, Hedral Inc.
#
#  Inputs:  A dataframe with a column named "text"
#
#  Output:  The same dataframe with the text changed to lowercase,
#           all noncharacter values and extra whitespace removed
#
#  Revision History:
#  AB 18NOV2020:  N/A, Initial Release.
#######################################################################

import pandas


def preprocess_mdr_text(df):

    # Set text to lower case, trim whitespace, and remove non-character
    # values, single character words, stopwords and multiple consecutive
    # spaces
    df['X'] = df['text'].str.lower().replace(
        "[^a-z]", ' ', regex=True
    ).replace(
        r"\s+[a-z]\s+", ' ', regex=True
    ).replace(
        r"\s+", ' ', regex=True
    ).str.strip()

    return df
