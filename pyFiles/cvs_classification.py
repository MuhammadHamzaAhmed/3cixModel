# Basics Libraries
import io
import os
import re
import time
import pickle

# For Data Handling
import numpy as np
import pandas as pd

# For Threading
from threading import Thread
from joblib import Parallel, delayed

# For PDF Text Extraction
from pdfminer3.pdfpage import PDFPage
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator, TextConverter

# For Text Handling
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from constants import *
from utilities import *



# Loads the CV Classification Model
def load_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model


# Extracts Raw Text from a PDF file
def extract_text_from_pdf_file(pdf_file_path):

    # pdfMiner Objects for PDF Text Extraction
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=(LAParams()))
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    # Reading the PDF File & Extracting its Text
    try:
        with open(pdf_file_path, 'rb') as (fh):
            num_pages = 0
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=False):
                page_interpreter.process_page(page)
                num_pages += 1

            text = fake_file_handle.getvalue()
        
        # Closing the PDF File Handlers
        converter.close()
        fake_file_handle.close()

        return text, num_pages
    
    except Exception as e:
        print('Error while Extracting Text from:', pdf_file_path)
        print('Error:', e)
        return ('', 0)


# Pre-Processes the Text Data
def preprocess_text(raw_text):

    try:

        if not raw_text:
            return ''
        
        # Removing Unwanted Characters
        text = raw_text.replace('\n', ' ')
        text = text.replace('\xa0', ' ')
        text = text.replace('\x0c', ' ')

        # Retaining only Alphabets
        text = re.sub('[^a-zA-Z]', ' ', text)
        main_words = text.lower().split()
        main_words = [w for w in main_words if len(w) > 1]


        # Getting Features
        num_words = len(main_words)
        num_chars = sum([len(w) for w in main_words])

        return num_words, num_chars

    except Exception as e:
        print('Error while Pre-Processing Text')
        print('Error:', e)
        return 0, 0


# Extracts CVs features
def extract_features(pdf_file_path):

    # Extracting Text
    raw_text, num_pages = extract_text_from_pdf_file(pdf_file_path)

    # Pre-Processing Text for Features
    num_words, num_chars = preprocess_text(raw_text)

    # Feature Engineering
    num_words_per_page = round(num_words / num_pages, 2) if num_pages > 0 else 0
    num_chars_per_page = round(num_chars / num_pages, 2) if num_pages > 0 else 0
    num_chars_per_word = round(num_chars / num_words, 2) if num_words > 0 else 0

    # Storing Features
    features = [num_pages, num_words, num_chars, num_words_per_page, num_chars_per_page, num_chars_per_word]

    # Converting to numpy
    features = np.array(features)

    return features


# Pre-Processing for CVs Classification
def pre_process(pdf_paths):
    features = Parallel(n_jobs=(-1), backend='multiprocessing')(map(delayed(extract_features), pdf_paths))
    features = np.array(features)
    return features


# Gets CV Classification Predictions
def get_predictions(pdf_paths):

    # Loading CV Classification Model
    model = load_model(clf_model_file)

    # Pre-Processing for Features
    features = pre_process(pdf_paths)

    # Getting Model Predictions
    predictions = model.predict(features)

    # Getting Predicted Labels
    prediction_labels = [cv_labels[pred] for pred in predictions]

    # Attaching Predicted Labels to PDF Files
    pdf_labels = dict(zip(pdf_paths, prediction_labels))

    return pdf_labels


# Returns PDF File Paths according to their Labels
def get_predicted_file_paths(pdf_labels):

    other_cv_paths, linkedin_cv_paths = [], []
    for pdf_file in pdf_labels:
        if pdf_labels[pdf_file] == 'Others':
            other_cv_paths.append(pdf_file)
        elif pdf_labels[pdf_file] == 'LinkedIn':
            linkedin_cv_paths.append(pdf_file)
        else:
            print('Got Unknown Classification Label')
            return

    return other_cv_paths, linkedin_cv_paths


# Copies the Classified PDF Files to their Respective Directories
def copy_classified_cvs(request_id, other_cv_paths, linkedin_cv_paths):
    
    if other_cv_paths:
        request_cvs_dir = os.path.join(downloaded_cvs_dir, request_id)
        for cv_path in other_cv_paths:
            target = os.path.join(request_cvs_dir, cv_path.split('/')[(-1)])
            copy_file(cv_path, target)

    if linkedin_cv_paths:
        request_linkedin_cvs_dir = os.path.join(downloaded_linkedin_cvs_dir, request_id)
        for cv_path in linkedin_cv_paths:
            target = os.path.join(request_linkedin_cvs_dir, cv_path.split('/')[(-1)])
            copy_file(cv_path, target)


# Separates S3 Links of CVs according to their Labels
def separate_classified_cvs(s3_cv_links, other_cv_paths, linkedin_cv_paths):
    
    other_cv_links, linkedin_cv_links = [], []
    for s3_cv_link in s3_cv_links:
        cv_link_filename = s3_cv_link.split('/')[(-1)]

        if other_cv_paths:
            for cv_path in other_cv_paths:
                if cv_link_filename == cv_path.split('/')[(-1)]:
                    other_cv_links.append(s3_cv_link)

        if linkedin_cv_paths:
            for cv_path in linkedin_cv_paths:
                if cv_link_filename == cv_path.split('/')[(-1)]:
                    linkedin_cv_links.append(s3_cv_link)

    return other_cv_links, linkedin_cv_links