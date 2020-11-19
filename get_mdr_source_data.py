#######################################################################
#  Title:   get_mdr_source_data.py
#
#  Purpose: Extract all Medical Device Reports (MDRs) from the openFDA
#           website for use in downstream processing.
#
#  Author:  Aaron Boussina, Hedral Inc.
#
#  Inputs:  openFDA Website
#
#  Output:  A Feather dataset (aeData.file) output to /SourceData
#           containing all available Medical Device Adverse Events,
#           the reported date, the device type, whether the event was
#           an Adverse Event, and the MDR text.
#
#  Revision History:
#  AB 18NOV2020:  N/A, Initial Release.
#######################################################################

# Import Packages
import subprocess
import pandas as pd
import json
import os

# Create Dictionary of downloads from FDA Downloads JSON
cmd = "curl 'https://api.fda.gov/download.json'"
downloads = json.loads(
    subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout
)

# Create List of Medical Device Adverse Event downloads
ae_downloads = downloads["results"]["device"]["event"]["partitions"]

#######################################################################
# Iterate through all Medical Device Adverse Event downloads
# and extract the information via curl commands.
# JSON return objects are flatted via list comprehension
# and a concatenated list of all events is created.
# The following values are preserved: the reported date,
# the device type, whether the event was an Adverse Event,
# and the MDR text.
#######################################################################

ae_list = list()
for ae_file in ae_downloads:
    print("Extracting Info From: " + ae_file['file'])
    cmd = "curl " + ae_file['file'] + " | zcat"
    ae_details = json.loads(
        subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout
    )["results"]

    ae_filtered = [
        (
            item["report_number"],
            item.get("report_date", ""),
            item.get("device", [{"generic_name": ""}])[0]["generic_name"],
            item["adverse_event_flag"],
            item.get("mdr_text", [{"text": ""}])[0]["text"],
        )
        for item in ae_details
        if "mdr_text" in item
        and len(item["mdr_text"]) > 0
        and item["mdr_text"][0]["text_type_code"]
        == "Description of Event or Problem"
        and item["adverse_event_flag"] != ""
    ]

    ae_list.extend(ae_filtered)


#######################################################################
# Convert the concatenated AE list into a dataframe and export the
# file in feather format to the SourceData folder
#######################################################################

ae_df = pd.DataFrame(ae_list)
ae_df.columns = ["docID", "date", "device_type", "aeYN", "text"]

ae_df.to_feather("SourceData/aeData.file")
