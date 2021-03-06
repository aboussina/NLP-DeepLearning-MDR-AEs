#######################################################################
#  Title:   plot_model_metrics.py
#
#  Purpose: Plot the Receiver Operating Characteristics curve and the
#           Precision Recall Curve for the model trained in
#           train_mdr_model.py to assess model performance.
#
#  Author:  Aaron Boussina, Hedral Inc.
#
#  Inputs:  The model parameters trained by train_mdr_model.py
#               1.  MdrModelStructure.json
#               2.  MdrModelWeights.h5
#
#           The dictionary of numpy arrays (mdrModelTestSet.npy)
#           for the test examples.
#
#  Output:  A plot of the Receiver Operating Characteristics Curve
#
#  Revision History:
#  AB 18NOV2020:  N/A, Initial Release.
#######################################################################

#######################################################################
# Import Packages, Source Data, and Model Parameters
#######################################################################

import numpy as np
from keras.models import model_from_json
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# Load model and model weights
src_model = "ModelOutputs/MdrModelStructure.json"
src_model_weights = "ModelOutputs/MdrModelWeights.h5"
src_test_set = "ModelOutputs/mdrModelTestSet.npy"

model_file = open(src_model)
model = model_from_json(model_file.read())
model.load_weights(src_model_weights)
model_file.close()

# Import Test Data Generated by getMdrSourceData.py
ae_data_test = np.load(src_test_set, allow_pickle=True)


#######################################################################
# Plot ROC Curve
#######################################################################

X_test = ae_data_test.item()['X']
y_test = ae_data_test.item()['y']
y_test_prediction = model.predict(X_test)


false_pos_rate, true_pos_rate, roc_thresh = roc_curve(
    y_test, y_test_prediction
)
auc = round(roc_auc_score(y_test, y_test_prediction), 3)
auc_label = "ROC Curve (area = " + str(auc) + ")"

plt.figure()
plt.plot(
    false_pos_rate,
    true_pos_rate,
    color="orangered",
    label=auc_label
)
plt.plot([0, 1], [0, 1], color="slateblue")
plt.axis([-0.004, 1, 0, 1.006])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve for MDR AE Classifier")
plt.legend(loc="lower right")
plt.savefig("Graphics/roc_curve.png")


#######################################################################
# Plot PRC Curve
####################################################################

precision, recall, prc_thresh = precision_recall_curve(
    y_test, y_test_prediction
)

plt.figure()
plt.plot(
    recall,
    precision,
    color="orangered"
)

plt.axis([-0.004, 1, 0, 1.008])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("PRC Curve for MDR AE Classifier")
plt.savefig("Graphics/prc_curve.png")


#######################################################################
# Retrieve Model Accuracy
####################################################################

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model_score = model.evaluate(X_test, y_test, verbose=1)
