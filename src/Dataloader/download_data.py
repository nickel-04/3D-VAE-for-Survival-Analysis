
import os
from nbiatoolkit import NBIAClient


def download_data():
    filePattern = '%PatientID/%StudyInstanceUID/%SeriesInstanceUID/%InstanceNumber.dcm'
    downloadDir = './NBIA-Download'
    nParallel = 10
    collection_name = "Colorectal-Liver-Metastases"


    # Check if the download directory already exists and is not empty
    if os.path.exists(downloadDir) and os.listdir(downloadDir):
        print(f"Data already downloaded in {downloadDir}. Skipping download.")
        return


    # Create the download directory if it does not exist
    os.makedirs(downloadDir, exist_ok=True)


    with NBIAClient(return_type="dataframe") as client:
        # Get all series for the specified collection
        series = client.getSeries(Collection=collection_name)
        unique_patient_ids = series['PatientID'].unique()


        # Download data for each unique patient
        for patient_id in unique_patient_ids:
            print(f"Downloading for Patient ID: {patient_id}")
            patient_series = series[series['PatientID'] == patient_id]


            for _, series_data in patient_series.iterrows():
                series_instance_uid = series_data['SeriesInstanceUID']
                print(f"Downloading Series: {series_instance_uid}")
                client.downloadSeries(
                    series_instance_uid,
                    downloadDir,
                    filePattern,
                    nParallel
                )


    print("Download complete!")


if __name__ == "__main__":
    download_data()


