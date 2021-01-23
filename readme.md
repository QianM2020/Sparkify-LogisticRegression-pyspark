 Sparkify User Churn Prediction
  (Data Scientist Capstone Project)

# Background

  Sparkify is a fictional music streaming app created by Udacity for this project.
   For this project we are given log data of this platform in order to drive insights and create a machine learning pipeline to predict churn.



# Getting Started
    Instructions below will help you setup your local machine to run the copy of this project. You can run the notebook in local mode on a local computer as well as on a cluster of a cloud provider such as AWS or IBM Cloud.

    # Prerequisites:

      Software and Data Requirements
      Anaconda 3
      Python 3.7.3
      pyspark 2.4
      pyspark.ml
      pandas

    # Dataset:

      The full dataset is 12GB, mini, medium and large datasets(only on AWS public) are available. I have used medium scale data that I have processed with Spark on AWS EMR.
      I used a mini-sized 128 MB dataset to process with Spark on Udacity Workspace.
      You can find a super mini-size dataset in a json file:'mini_sparkify_event_data.json',
      to drive insights of the data.

      If you have an AWS account, a large dataset(12 GB) has been public on s3n://udacity-dsnd/sparkify/sparkify_event_data.json

    # Running the notebooks

      Install all the packages stated above.
      Run the commands below in your working directory to open the project in jupyter lab:
        git clone https://github.com/QianM2020/Sparkify.git
        jupyter lab

      Sparkify.ipynb: This notebook has all the functions for processing the data and ml pipeline.

    If you have difficulty in displaying .ipynb files please go to https://nbviewer.jupyter.org/ and paste the link that you're trying to display the notebook. And you can use http://htmlpreview.github.io/ to display html files.

    # Summary and Report
      you can find more details about analysis and discussion about the project in 'Report.docx'
      or on CSDN Blog Post: https://blog.csdn.net/QianMeng2020/article/details/113027411
