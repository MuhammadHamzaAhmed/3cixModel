import os
import json
import time
import boto3
import shutil
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from threading import Thread
from joblib import Parallel, delayed
from botocore.exceptions import NoCredentialsError
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


from pyFiles.constants import *


# Creates the required Download Directories
def create_dirs():
    try:
        os.makedirs(downloaded_cvs_dir, exist_ok=True)
        os.makedirs(downloaded_linkedin_cvs_dir, exist_ok=True)
        os.makedirs(downloaded_clf_cvs_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(requests_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
    except Exception as e:
        print('Error while Creating Directories')
        print('Error:', e)
        return


# Creates a Directory for the Request
def create_request_cvs_dir(request_id):
    try:
        request_cvs_dir = os.path.join(downloaded_cvs_dir, request_id)
        request_linkedin_cvs_dir = os.path.join(downloaded_linkedin_cvs_dir, request_id)
        request_clf_cvs_dir = os.path.join(downloaded_clf_cvs_dir, request_id)
        os.makedirs(request_cvs_dir, exist_ok=True)
        os.makedirs(request_linkedin_cvs_dir, exist_ok=True)
        os.makedirs(request_clf_cvs_dir, exist_ok=True)
    except Exception as e:
        print('Error while Creating directory for Request:', request_id)
        print('Error:', e)
        return


# Copies File from 'source' to 'target'
def copy_file(source, target):
    try:
        shutil.copyfile(source, target)
    except shutil.SameFileError:
        print("Source: '{}' and Target: '{}' are the same file.".format(source, target))
    except IsADirectoryError:
        print("Target: '{}' is a Directory.".format(target))
    except PermissionError:
        print("Permission Denied while Copying Source: '{}' to Target: '{}'".format(source, target))
    except:
        print("Error occurred while Copying Source: '{}' to Target: '{}'".format(source, target))


# Deletes all the files within Directories
def delete_files(request_id):
    try:
        request_cvs_dir = os.path.join(downloaded_cvs_dir, request_id)
        if os.path.exists(request_cvs_dir):
            for cv_file in os.listdir(request_cvs_dir):
                os.remove(os.path.join(request_cvs_dir, cv_file))
        request_linkedin_cvs_dir = os.path.join(downloaded_linkedin_cvs_dir, request_id)
        if os.path.exists(request_linkedin_cvs_dir):
            for cv_file in os.listdir(request_linkedin_cvs_dir):
                os.remove(os.path.join(request_linkedin_cvs_dir, cv_file))
        request_clf_cvs_dir = os.path.join(downloaded_clf_cvs_dir, request_id)
        if os.path.exists(request_clf_cvs_dir):
            for cv_file in os.listdir(request_clf_cvs_dir):
                os.remove(os.path.join(request_clf_cvs_dir, cv_file))
    except Exception as e:
        print('Error while Deleting Files from Directories')
        print('Error:', e)
        return


# Deletes all Old Files (For Debugging)
def delete_all_old_files():
    try:
        shutil.rmtree(downloaded_cvs_dir, ignore_errors=True)
        shutil.rmtree(downloaded_linkedin_cvs_dir, ignore_errors=True)
        shutil.rmtree(downloaded_clf_cvs_dir, ignore_errors=True)
        shutil.rmtree(results_dir, ignore_errors=True)
        shutil.rmtree(requests_dir, ignore_errors=True)
        shutil.rmtree(logs_dir, ignore_errors=True)
    except Exception as e:
        print('Error while Deleting Old Files')
        print('Error:', e)
        return


# Downloads & saves the file from its URL
def download_file(download_dir, cv_download_url):
    try:
        filename = cv_download_url.split('/')[(-1)]
        filename = os.path.join(download_dir, filename)

        # Downloading with retries
        s = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        s.mount('https://', HTTPAdapter(max_retries=retries))
        response = s.get(cv_download_url)

        # Writing to file
        file = open(filename, 'wb')
        file.write(response.content)
        file.close()

    except Exception as e:
        print('Error while Downloading File:', filename)
        print('Error:', e)
        return


# Downloads all the files from their URLs
def download_files(cv_download_urls, download_dir, multithreading=True):

    if multithreading:
        Parallel(n_jobs=-1, backend='multiprocessing')(map(delayed(download_file), [download_dir for _ in cv_download_urls], cv_download_urls))
    else:
        for cv_download_url in cv_download_urls:
            download_file(download_dir, cv_download_url)


# Uploads File to S3
def upload_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Results Uploaded Successfully")
        return True
    except FileNotFoundError:
        print("Results File Not Found")
        return False
    except NoCredentialsError:
        print("Wrong Credentials")
        return False


# Delete File from S3
def delete_from_s3(bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    try:
        s3.delete_object(bucket, s3_file)
        print("Previous Results File Deleted")
        return True
    except FileNotFoundError:
        print("Results File Not Found")
        return False
    except NoCredentialsError:
        print("Wrong Credentials")
        return False


# Posts Results to API
def post_results(api_endpoint, results_dict):
    response = ''
    try:
        response = requests.post(url=api_endpoint, data=results_dict, verify=True)
        if '200' in str(response):
            return True, None
        return False, response
    except Exception as e:
        print('Error while Sending Results to POST API')
        print('Error:', e)
        return False, response


# Removes NaNs or None
def remove_nans_from_dict(ranked_cvs_dict, nans_replace_str):
    for cv_dict in ranked_cvs_dict:
        for k in cv_dict:
            if (type(cv_dict[k]) in [float, int]) and (np.isnan(cv_dict[k])):
                cv_dict[k] = nans_replace_str
            if k == 'Total Experience (months)' and cv_dict[k] is None:
                cv_dict[k] = 0
            elif cv_dict[k] is None:
                cv_dict[k] = nans_replace_str
    return ranked_cvs_dict


# Checks if Results are obtained
def check_results_obtained(cvs, linkedin_cvs):
    cvs_obtained = False
    linkedin_cvs_obtained = False
    if not is_nan(cvs):
        cvs_obtained = True
    if not is_nan(linkedin_cvs):
        linkedin_cvs_obtained = True
    return cvs_obtained, linkedin_cvs_obtained


# Returns Today's Datetime
def get_current_datetime():
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
    return dt_string


# Returns a Logger
def create_logger(request_id, current_datetime):

    log_filename = logs_filename + '_' + request_id + '_' + current_datetime + '.log'
    log_file = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger("API Logger - Request: " + request_id)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


# Writes Dictionary to JSON File
def write_to_json(file_dir, filename, request_id, current_datetime, dict_to_write):
    file_name = filename + '_' + request_id + '_' + current_datetime + '.json'
    file_name = os.path.join(file_dir, file_name)
    with open(file_name, 'w') as outfile:
        json.dump(dict_to_write, outfile)


# Checks for NULL Values
def is_nan(s):
    if type(s) is pd.DataFrame:
        nans_cond = (s is None) or \
                    (s.empty) or \
                    (len(s) == 0)
    else:
        nans_cond = (not s) or \
                    (s is None) or \
                    (s == 'nan') or \
                    (s == 'None') or \
                    (type(s) is float and np.isnan(s)) or \
                    (type(s) is list and not s) or \
                    (type(s) is list and len(s) == 1 and s[0] is None)
    return nans_cond


# Returns Length of Array/Dataframe
def get_length(a):
    if is_nan(a):
        return 0
    else:
        return len(a)



# Class for Threading with Return Value
class ReturnThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return