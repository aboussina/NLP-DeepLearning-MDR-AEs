#######################################################################
#  Title:   preprocess_mdr_text.py
#
#  Purpose: Extract all Medical Device Reports (MDRs) from the openFDA
#           website for use in downstream processing.
#
#  Author:  Aaron Boussina, Hedral Inc.
#
#  Inputs:  openFDA Website
#
#  Output:  An hdf5 dataset (aeData.h5) output to the current directory
#           containing all available Medical Device Adverse Events,
#           the reported date,the device type, whether the event was
#           an Adverse Event, and the MDR text.
#
#  Revision History:
#  AB 14NOV2020:  N/A, Initial Release.
#######################################################################

from nltk.corpus import stopwords
import pandas


def preprocess_mdr_text(df):
    # Create a regex query to remove all stop words
    swList = stopwords.words("english")
    stopWordQuery = ''
    for word in swList:
        stopWordQuery = (
            stopWordQuery + "|" + r"\s+" + word + r"\s+" + "|"
            + "^" + word + r"\s+" + "|" + r"\s+" + word + "$"
        )

    stopWordQuery = stopWordQuery[1:]

    # Set text to lower case, trim whitespace, and remove non-character
    # values, single character words, stopwords and multiple consecutive
    # spaces
    df['X'] = df['text'].str.lower().replace(
        "[^a-z]", ' ', regex=True
    ).replace(
        r"\s+[a-z]\s+", ' ', regex=True
    ).replace(
        stopWordQuery, ' ', regex=True
    ).replace(
        r"\s+", ' ', regex=True
    ).str.strip()

    return df
