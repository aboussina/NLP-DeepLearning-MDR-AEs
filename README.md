# Prediction of Medical Device Report Adverse Events based on event description
## Abstract
Post market surveillance is a regulatory requirement for medical device manufacturers per 21 CFR Part 822.  To meet this requirement, device manufacturers employ complaint specialists to handle incoming events and classify them as Adverse Events (AEs) or not.  A subset of these complaints (AEs or those with the potential to have been AEs) are submitted to the FDA and captured in the MAUDE (Manufacturer and User Facility Device Experience) database.  The process for handling these complaints is typically manual, with complaint specialists fielding many incoming events per day in a first in, first out (FIFO) manner.  To explore whether deep learning could be utilized to automate triaging for complaint response, a binary classification neural network was developed using all events (~4.8M) in the MAUDE database.  On a testing set of approximately ~3.84M records, the trained model achieved 93.68% accuracy and an AUROC of 0.984 for predicting the occurence of an adverse event based on the event description and the device type.
<br/>
<br/>
FDA Disclaimer on MDR data: "Although MDRs are a valuable source of information, this passive surveillance system has limitations, including the potential submission of incomplete, inaccurate, untimely, unverified, or biased data. In addition, the incidence or prevalence of an event cannot be determined from this reporting system alone due to potential under-reporting of events and lack of information about frequency of device use. Because of this, MDRs comprise only one of the FDA's several important postmarket surveillance data sources." 
<br/>
<br/>

## Methods & Results
### Source Data
Events in the MAUDE database were accessed through openFDA (https://open.fda.gov).  Since the openFDA API limits queries to 1000 at a time, the files were downloaded with get_mdr_source_data.py by performing curl commands on all relevant downloads from https://api.fda.gov/download.json and flattening the resulting JSON responses into a pandas dataframe.  The fields of interest were "adverse_event_flag", "mdr_text.text" [when "mdr_text.text_type_code" == "Description of Event or Problem"] and "device.generic_name".  Events with missing values for "adverse_event_flag" were excluded from the output dataset.
<br/>
<br/>

### Preprocessing & Model
Standard text preprocessing was performed (preprocess_mdr_text.py) and MDR text was set to lowercase, all noncharacter values and single character words were removed, and extra whitespace was trimmed.  The data was then split into training (80%) and test (20%) sets and the processed MDR text was tokenized with tensorflow.keras.preprocessing.text.  The Tokenizer word index was created on the training set and the corresponding words were converted to numeric sequences and padded to the longest sentence length.  A dense neural network model was then created with an embedding layer and three hidden layers using Keras.  The Adam optimizer was used with binary cross entropy loss to fit the model to the training data.
<br/>
<br/>

### Results
[!ROC Curve](Graphics/roc_curve.png)


[!PRC Curve](Graphics/prc_curve.png)

## How to Run
The model structure, tokenizer, and test set are in the /ModelOutputs directory.  The larger files (aeFile.file.xz and MdrModelWeights.h5.xz) can be downloaded from https://drive.google.com/drive/folders/1pOyKLxE4jjsqskz_ljVFJSNfxsWbSGmJ?usp=sharing.  Decompress those files and place aeFile.file into the /SourceData directory and MdrModelWeights.h5 into the /ModelOutputs directiory.  Then, to test specific events, enter the events into the enterAEs.csv spreadsheet and run predict_mdr_ae.py to get a terminal printout of the model's predictions.  
<br/>
<br/>
Alternatively, the model can be built by running:
1. get_mdr_source_data.py to download the entirety of MAUDE events.
2. train_mdr_model.py to train the neural network.
3. plot_model_metrics.py to visualize performance.
<br/>
<br/>

## Next Steps
Due to hardware limitations, LSTMs and other RNNs could not be trained on the whole training set.  Testing of these models on a subset of the training data showed some promise for achieving higher accuracies (90.11% accuracy for 50,000 training examples).  Future efforts will therefore focus on alternative neural network architectures as well as hyperparameter tuning.

