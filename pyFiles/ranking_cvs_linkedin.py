# Basics Libraries
import io
import os
import re
import time
import pickle

# For Data Handling
import numpy as np
import pandas as pd
from collections import Counter

# For Handling Text
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz


# Importing Files
# from constants import *
# from data import *



#### Pre-Processing JDs


# Returns the Experience in number of months
def get_experience_months(experience_string, range_num_months=12):
    if type(experience_string) is str:
        if '-' in experience_string:
            num_years, year_str = experience_string.split()
            min_years, max_years = num_years.split('-')
            return int(((int(min_years) + int(max_years)) / 2) * range_num_months)
        else:
            num_years, year_str = experience_string.split()
            return int(num_years) * 12
    else:
        if type(experience_string) is int:
            return experience_string * 12
        print("Unexpected Type for 'Required experience'")


# Adds column of Experience in number of months 
def add_num_months_experience_column(jds_df):
    jds_df.insert(jds_df.columns.get_loc('Required Experience') + 1, 'Average No. of Months Experience', jds_df['Required Experience'].apply(get_experience_months))



### Ranking


# Returns the Matching Percentage for two Strings
def get_string_matching(str1, str2, partial=True):
    if (type(str1) is str) and (type(str2) is str) and str1 and str2:
        if partial:
            matching_percentage = round(fuzz.partial_ratio(str1, str2))
        else:
            matching_percentage = round(fuzz.ratio(str1, str2))
        return matching_percentage
    else:
        print('Unexpected Variable type for string Matching')
        return


# Returns Dataframe of CVs with closest Experiences
def get_cvs_closest_experiences(cvs_df, required_experience_months, experience_range_threshold = 12):
    
    if required_experience_months < 12:
        print('Unexpected Experience Required')
        return
    
    # Difference in Experience
    # cvs_df['Experience Difference'] = np.abs(cvs_df['Total Experience (months)'] - required_experience_months)
    cvs_df['Experience Difference'] = cvs_df['Total Experience (months)'] - required_experience_months
    
    # CVs within Experience Range
    # closest_cvs = cvs_df[cvs_df['Experience Difference'] <= experience_range_threshold].copy(deep=True)
    closest_cvs = cvs_df[cvs_df['Experience Difference'] >= 0].copy(deep=True)
    closest_cvs.reset_index(drop=True, inplace=True)

    # Sorting CVs with closest Experience
    # closest_cvs = closest_cvs.sort_values('Experience Difference')
    
    # Removing Un-needed Column
    closest_cvs.pop('Experience Difference')
    
    return closest_cvs


# Returns Dataframe of CVs with similar Location
def get_closest_location_cvs(cvs_df, location, matching_ratio = 80):
    
    # Matching 'Location' by partial string matching
    cvs_df['Location Matched'] = cvs_df.apply(lambda cv: get_string_matching(cv['Location'], location), axis=1) > matching_ratio
    
    # CVs with Matched Location
    closest_cvs = cvs_df[cvs_df['Location Matched'] == True].copy(deep=True)
    closest_cvs.reset_index(drop=True, inplace=True)

    cvs_df.pop('Location Matched')
    
    return closest_cvs

 
# Returns Dataframe of CVs with similar Job Titles
def get_closest_job_title_cvs(cvs_df, job_title, matching_ratio = 80):
    
    # Matching 'Location' by partial string matching
    cvs_df['Domain Matched'] = cvs_df.apply(lambda cv: get_string_matching(cv['Domain'], job_title), axis=1) > matching_ratio
    
    # CVs with Matched Location
    closest_cvs = cvs_df[cvs_df['Domain Matched'] == True].copy(deep=True)
    closest_cvs.reset_index(drop=True, inplace=True)

    cvs_df.pop('Domain Matched')
    
    return closest_cvs


# Pre-Processes the Text Data
def preprocess_text(text, remove_stop_words = True, lemmatization = True):

    # Retaining only the Alphabets
    main_words = re.sub('[^a-zA-Z]', ' ', str(text))
    main_words = main_words.lower().split()
    
    # Removing stopwords
    if remove_stop_words:                                           
        main_words = [w for w in main_words if not w in set(stopwords.words('english'))]

    # Lemmatization: Grouping different forms of the same word
    if lemmatization:
        main_words = [WordNetLemmatizer().lemmatize(w) for w in main_words if len(w) > 1]

    # Removing letters
    main_words = [w for w in main_words if len(w) > 1]
        
    return main_words


# Checks for NULL Types
def is_nan(s):
    return s is None or s == 'nan' or type(s) is float and np.isnan(s)


# Returns the required text from JD
def get_jd_text(jd):
    
    # Combining JDs Text
    jd_text = ''
    jd_text += jd['Industry'] + ' ' if not is_nan(jd['Industry']) else ''
    jd_text += jd['Soft Skills'] + ' ' if not is_nan(jd['Soft Skills']) else ''
    jd_text += jd['Job Title'] + ' ' if not is_nan(jd['Job Title']) else ''
    jd_text += jd['Tools Handling Experience'] + ' ' if not is_nan(jd['Tools Handling Experience']) else ''
    jd_text += jd['Mandatory Requirement'] + ' ' if not is_nan(jd['Mandatory Requirement']) else ''
    jd_text += jd['Certification'] + ' ' if not is_nan(jd['Certification']) else ''
    jd_text += jd['Detailed Job Description'] + ' ' if not is_nan(jd['Detailed Job Description']) else ''
    
    # Cleaning Text
    jd_text = jd_text.replace('\xa0', ' ')
    jd_words = preprocess_text(jd_text)
    sorted_jd_words = sorted(jd_words)
    jd_words = Counter(sorted_jd_words)
    
    return jd_words


# Returns the required text from CV
def get_cv_text(cv):

    # ['Name', 'Location', 'Domain', 'Summary', 'Experience', 'Total Experience (months)', 'Education', 'S3 Path']
    _, _, domain, summary, experience, _, _, _ = cv
    
    # Combining CVs Text
    cv_text = ''
    cv_text += domain + ' ' if not is_nan(domain) else ''
    cv_text += summary + ' ' if not is_nan(summary) else ''
    
    if (type(experience) is list) and (len(experience) > 0) and (type(experience[0]) is dict):
        cv_exp = ''
        for exp in experience:
            if 'Designation' in exp:
                cv_exp += exp['Designation'] + ' ' if not is_nan(exp['Designation']) else ''
            else:
                cv_exp += exp['Designation/Organization'] + ' ' if not is_nan(exp['Designation/Organization']) else ''
            cv_exp += exp['Description'] + ' ' if not is_nan(exp['Description']) else ''
        cv_text += cv_exp
        
    elif (type(experience) is list) and (len(experience) > 0) and (type(experience[0]) is str):
        cv_exp = ' '.join(cv['Experience'])
        

    # Cleaning Text
    cv_text = cv_text.replace('\xa0', ' ')
    cv_words = preprocess_text(cv_text)
    sorted_cv_words = sorted(cv_words)
    cv_words = Counter(sorted_cv_words)
    
    return cv_words


# Get Similarity Count between JD & CV, based on common words
def get_jd_cv_similarity(jd_words, cv_words):
        
    word_count = 0
    for cv_word in cv_words:
        if cv_word in jd_words:
            word_count += cv_words[cv_word]
    return word_count


# Ranks CVs according to JD
def rank_cvs(jd, cvs_df, check_location = False, check_job_title = False, check_required_experience_months = False):
    
    if type(jd) in [dict, pd.Series, pd.DataFrame]:
    
        # Getting required text of JD
        jd_text = get_jd_text(jd)

        closest_cvs = cvs_df

        # Getting similar Location CVs
        if check_location:
            closest_cvs = get_closest_location_cvs(cvs_df, jd['Location'])

        # Getting similar Job Title CVs
        if check_job_title:
            closest_cvs = get_closest_job_title_cvs(closest_cvs, jd['Job Title'])

        # Getting Closest Experience CVs
        if check_required_experience_months:

            # Required Experience in Months
            required_experience_months = get_experience_months(jd['Required Experience'])

            # Getting Closest Experience CVs
            closest_cvs = get_cvs_closest_experiences(closest_cvs, required_experience_months)

        # Getting Similarity Scores
        sims = []
        for cv in closest_cvs.itertuples(index=False):
            cv_text = get_cv_text(cv)
            word_count = get_jd_cv_similarity(jd_text, cv_text)
            sims.append(word_count)

        closest_cvs['Similarity Score'] = sims

        # Sorting CVs with Similarity Score
        closest_cvs = closest_cvs.sort_values('Similarity Score', ascending=False)

        # Resetting Dataframe Index
        closest_cvs.reset_index(drop=True, inplace=True)
    
        return closest_cvs
    
    else:
        print('Unexpected type for Job Description (JD)')
        return
