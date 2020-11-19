#######################################################################
#  Title:   predict_mdr_ae.py
#
#  Purpose: Predict whether an MDR event is an adverse event based on
#           the event description.
#
#  Author:  Aaron Boussina, Hedral Inc.
#
#  Inputs:  The model parameters trained by train_mdr_model.py
#               1.  MdrModelStructure.json
#               2.  MdrModelWeights.h5
#               3.  mdrToken.json
#
#           A csv containing event description(s) and device type(s)
#           (enterAE.csv)
#
#  Output:  A printout to the console log of whether the event is
#           adverse or not.
#
#  Revision History:
#  AB 18NOV2020:  N/A, Initial Release.
#######################################################################


#######################################################################
# Import Packages, Source Data, and Model Parameters
#######################################################################

import pandas as pd
from preprocess_mdr_text import *
from keras.models import model_from_json
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import CSV containing events as dataframe
src_file = "SourceData/enterAE.csv"
ae_predict = pd.read_csv(src_file)

# Load model, model weights, and model tokenizer
src_model = "ModelOutputs/MdrModelStructure.json"
src_model_weights = "ModelOutputs/MdrModelWeights.h5"
src_tokenizer = "ModelOutputs/mdrToken.json"

model_file = open(src_model)
model = model_from_json(model_file.read())
model.load_weights(src_model_weights)
model_file.close()

tokenFile = open(src_tokenizer)
mdr_token = tokenizer_from_json(tokenFile.read())
tokenFile.close()


#######################################################################
# Text Preprocessing and Tokenization
#######################################################################

ae_predict['text'] = ae_predict['device_type'] + ' ' + ae_predict['text']
preprocess_mdr_text(ae_predict)

X_predict = mdr_token.texts_to_sequences(ae_predict['X'])

padding_length = model.input.shape.as_list()[1]
X_predict = pad_sequences(X_predict, maxlen=padding_length, padding='post')


#######################################################################
# Output Prediction
#######################################################################

prediction = model.predict(X_predict)

for i in range(len(prediction)):
    if prediction[i][0] > 0.5:
        ae = 'Yes'
    else:
        ae = 'No'

    print("\n")
    print("Event Description: " + ae_predict['X'][i])
    print("Adverse Event: " + ae)
    print("Probability: " + str(prediction[i][0]))
    print("\n")
