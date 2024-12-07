
## DSA5208 Project 2: Predicting Air Pressure with MLlib (Apache Spark)
S K Ruban A0253837W, Owen Li Dong Lin A0231088H

## Description and Tasks

- Use MLlib of Apache Spark to build a machine learning model that predicts the air pressure based on geographical parameters and weather conditions.
- Download the dataset from https://www.ncei.noaa.gov/data/global-hourly/archive/csv/2023.tar.gz, whose documentation can be found at https://www.ncei.noaa.gov/data/global-hourly/doc/.
- Build machine learning models to predict the sea level pressure by using the following data:
[Latitude coordinates, Longitude coordinates, Elevation, Wind direction angle, Wind speed rate, Ceiling height, Visibility distance, Air temperature, Dew point temperature]
- Note that the csv files contain missing data, which must be removed before training. The missing data are denoted using some special values described in the documentation.
- During training, set the ratio of the training set and test set to 7:3. Use the validation tools provided by MLlib to find the best model. Print the training error and the test error on screen.
- You can use cloud computing services such as Amazon AWS and Google Cloud to run your codes.

## Deliverables

- This is again a group project and each group should include no more than 3 people. When finishing the project, submit the following:
1. A document including a brief description of the method and the test results.
2. The code and a README file.
3. Your conversation with any AI tools (such as ChatGPT) that assisted you in completing the project, if applicable.
Please submit these documents to Canvas no later than 17 November, 2024.

---

## Set-up overview
This project involves the extraction and upload of weather-related data from a Google Cloud Storage (GCS) bucket to a Google Cloud DataProc cluster. The data, provided in a .tar.gz format, is extracted and processed, and the individual CSV files are re-uploaded to the GCS bucket for further analysis. 

### **Prerequisites**

- **Google Cloud Platform** account.
- **Google Cloud SDK** installed and configured on your local machine.
- **Google Cloud Bucket**: create a GCS bucket with a custom name.
- **DataProc Cluster**: A cluster named `project-cluster` with a master node called `project-cluster-m`.

### **Setup steps**

We will start with downloading, extracting and saving the data.

First, download the 2023.tar.gz file on your local machine. Next, follow the steps below to upload the file to your google cloud bucket and process in the cloud.

#### 1. SSH into the master node
gcloud compute ssh extract-cluster-m

#### 2. Create directory
mkdir weather_data
cd weather_data

#### 3. Copy and extract
gsutil cp gs://your-bucket-name/2023.tar.gz .
tar -xzf 2023.tar.gz

#### 4. Upload extracted files
gsutil -m cp -r *.csv gs://your-bucket-name/extracted/

#### 5. Cleanup
cd ..
rm -rf weather_data

### **Load saved notebook and run code**
Now, the data is ready for use. 

Proceed to the 'Dataproc' page on the Google Cloud web interface. Enter the cluster-specific page by clicking the relevant cluster name. 

Then, enter 'Web Interfaces' by clicking on the header name. Under 'Component gateway', open a jupyter notebook by clicking 'Jupyter'.

Now, load the .ipynb file that we submitted. After changing the relevant GCS bucket names, directories and potentially cluster names (if different), the code can be run in the notebook!
