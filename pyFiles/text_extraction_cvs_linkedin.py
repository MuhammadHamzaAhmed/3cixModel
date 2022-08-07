# Basics Libraries
import io
import os
import re
import time
from datetime import date, datetime
from joblib import Parallel, delayed

# For Data Handling
import numpy as np
import pandas as pd

# For PDF Text Extraction
from pdfminer3.pdfpage import PDFPage
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator, TextConverter

# Importing Constants
from pyFiles.constants import *

# Extracts Raw Text from a PDF file
def extract_text_from_pdf_file(pdf_file_path):
    
    # pdfMiner Objects for PDF Text Extraction
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
    try:

        # Reading the PDF File & Extracting its Text
        with open(pdf_file_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()
            
        # Closing the PDF File Handlers
        converter.close()
        fake_file_handle.close()

        return text


    except Exception as e:
        print('Error while Extracting Text from:', pdf_file_path)
        print('Error:', e)
        return


# Extracts Raw Text from multiple PDF files
def extract_text_from_pdf_dir(pdf_dir_path):
    
    # pdfMiner Objects for PDF Text Extraction
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
    try:
        
        # For all Files in the Directory
        all_pdfs_text = ''
        for pdf_file in os.listdir(pdf_dir_path):

            # Checking for PDF files
            if pdf_file.lower().endswith('.pdf'):

                # Path to the CV (PDF) file
                pdf_file_path = os.path.join(pdf_dir_path, pdf_file)

                # Reading the PDF File & Extracting its Text
                with open(pdf_file_path, 'rb') as fh:
                    for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                        page_interpreter.process_page(page)
                    text = fake_file_handle.getvalue()
                    
                # Storing all Raw Text
                all_pdfs_text += text + next_cv_string

        # Closing the PDF File Handlers
        converter.close()
        fake_file_handle.close()
        
        return all_pdfs_text


    except Exception as e:
        print('Error while Processing PDFs Directory:', pdf_dir_path)
        print('Error:', e)
        return



### Text Pre-Processing


# Inserts PDF Page Separator
def insert_pdf_page_separator(text):
    pages_text = []
    for t in text:
        pages_text.append(t)
        pages_text.append(pdf_page_separator)
    return pages_text


# Cleans the Raw Text
def clean_text(raw_text):
    
    # Seperating PDF Pages
    text = raw_text.split(next_cv_string)
    text = insert_pdf_page_separator(text)
    
    # Removing Unwanted Characters
    text = [t.replace('\xa0', '') for t in text]
    text = [t.replace('\x0c', '') for t in text]
    text = [s for t in text for s in t.split('\n')]
    
    # Removing Empty Strings
    cleaned_text = [t for t in text if t]

    return cleaned_text


# Checks for Duplication of Fields (for separation of multiple CVs)
def check_duplicates(text, cv_ident):
    
    # Checking for the second occurence of the cv_ident
    num_found = 0
    for i in range(len(text)):
        if cv_ident == text[i:i + len(cv_ident)]:
            num_found += 1
        if num_found > 1:
            return i
    return None


# Checks wether a string is a Name
def is_name(name):
    names = name.split()
    if len(names) in [2, 3] and names[0].isalpha() and names[1].isalpha():
        return bool(re.match(r"([A-Z][a-z][a-z]*) *[A-Z]* *([A-Z][a-z][a-z][a-z]*)", name))
    else:
        return False

    
# Returns the correct cv_ident_len
def get_cv_ident_len(text):
    
    # Check for Next Section (within 5 lines)
    for i in range(7):
        if text[i] in ['Experience', 'Summary', pdf_page_separator]:
            return i

    return -1


# Splits individual CVs from a PDF conatining multiple CVs
def split_cvs(text):
    
    # Splitting CVs
    cvs = []
    num_iters, max_num_iters = 0, 1000    # For Handling Infintie Loop
    i = 0
    while True:
        
        # Picking the CV Name, Location & Position
        if i == 0:
            cv_ident_len = get_cv_ident_len(text[i:])
            cv_ident = text[i: i + cv_ident_len]
        
        elif (i < len(text)) and (i + cv_ident_len < len(text)):
            cv_ident_len = get_cv_ident_len(text[i + 1:])
            cv_ident = text[i + 1: i + cv_ident_len + 1]
        
        else:
            cv_ident = text[i + 1::]
            if cv_ident == [pdf_page_separator]:
                break
            
        # Checking for duplicate occurence of CV Name, Location & Position
        duplicate_index = check_duplicates(text, cv_ident)
        if duplicate_index:
#             print(i, duplicate_index)
#             print(text[i], text[duplicate_index])
            p = i
            i = duplicate_index + cv_ident_len
#             print('\n\n')
#             print('cv_ident:', cv_ident)
#             print('P:', p, 'I:', i, 'LEN:', len(text))
#             print(text[p:i - cv_ident_len])
#             print(text[i])
            if text[i] == pdf_page_separator:
                if i >= len(text):
                    cv_details = [t for t in text[p:len(text) - cv_ident_len] if t != pdf_page_separator]
                    cvs.append(cv_details)
                    break
                else:
                    cv_details = [t for t in text[p:i - cv_ident_len] if t != pdf_page_separator]
                    cvs.append(cv_details)
            
            # To Skip LinkedIn Recommendations
            elif (linkedin_recommendation_string in text[i]) or (not is_name(text[i])):
                j = i
                while True:
                    if text[j] == pdf_page_separator:
                        if j + 1 == len(text):
                            return cvs
                        if is_name(text[j + 1]):
                            i = j
                            break
                    j += 1
            else:
                # print('Issue in Parsing CVs: Unable to find end of CV')
                return
        else:
            # print('Issue in Parsing CVs: Unable to find duplicate cv_ident')
            return
        
        # Handling Infinite Loop
        num_iters += 1
        if num_iters >= max_num_iters:
            # print('Issue in Parsing CVs: Max Number of Iterations reached')
            return
        
    return cvs



# Parses the 'Personal Info.' from CV
def parse_personal_info(cv):
    
    # Getting Index of Summary/Experience
    i = 0
    while i < len(cv):
        if cv[i] in ['Experience', 'Summary', pdf_page_separator]:
            break
        i += 1
    
    # Returning the cv_ident according to length
    if i == 3:
        return cv[:i]
    
    elif i > 3:
        cv_ident = cv[:2]
        cv_position = ''
        for i in range(2, i):
            cv_position += cv[i] + ' '
        cv_ident.append(cv_position)
        return cv_ident
        
    else:
        # print('Unexpected Length of cv_ident')
        return None
    

# Parses the 'Summary' field from CV
def parse_summary(summary_list):
    
    summary_description = ''
    for sd in summary_list[1::]:
        summary_description += sd
                
    return summary_description
    
# Parses the 'Education' field from CV
def parse_education(education_list):
    
    try:
        education_dict = {}
        education_dict['Institute'] = education_list[1]

        # Getting Education Year
        for e in education_list:
            year = re.findall(r"([1-2][0-9][0-9][0-9] *- *[1-2][0-9][0-9][0-9])", e)
            if year:
                education_dict['Years'] =  year[0]
                break

        return education_dict
    
    except Exception as e:
        # print('Error:', e)
        return None

# Parses the 'Experience' field from CV
def parse_experience(experience_list):

    try:
        experience_found = False
        experiences_found = []
        for i, e in enumerate(experience_list):

            is_present = False
            if (present_string in e):
                if (len(e.split()) > 3) and (e.split()[3] == present_string):
                    is_present = True
            
            # Matching Date e.g: "January 2018 - Present"
            if is_present:
                date_match = r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May?|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) (?:19[7-9]\d|2\d{3})(?=\D|$)"
                matched_dates = re.findall(date_match, e)
                
            # Matching Date e.g: "January 2012 - February 2016"
            else:
                date_match = r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May?|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) (?:19[7-9]\d|2\d{3})(?=\D|$)"
                dates_match = date_match + " *- *" + date_match
                matched_dates = re.findall(dates_match, e)

            # Matching Duration e.g: "(4 years 2 months)"
            year_match = re.findall(r"([1-9][0-9]* *year(?:s)?)", e)
            month_match = re.findall(r"([1-9][0-9]* *month(?:s)?)", e)

            if matched_dates and (year_match or month_match):
                experience_found = True
                
                # Adding Experience Details
                if is_present:
                    start_date, end_date = matched_dates[0], present_string
                else:
                    start_date, end_date = matched_dates[0].split(' - ')
                
                num_years = int(year_match[0].split()[0]) if year_match else 0
                num_months = int(month_match[0].split()[0]) if month_match else 0
                    
                experiences_found.append([i, start_date, end_date, num_years, num_months])
        
        # Experience Dates not Found
        if not experience_found:
            if (len(experience_list[1::]) == 1) and (' at ' in experience_list[1::][0]):
                experience_details_dict = dict()
                designation_organization = experience_list[1::][0].split(' at ')
                experience_details_dict['Designation'] = designation_organization[0]
                experience_details_dict['Organization'] = designation_organization[1]
                return experience_details_dict
            else:
                return experience_list[1::]
                
        experience_details = []
        for i, ed in enumerate(experiences_found):
            experience_details_dict = dict()
            
            if ' at ' in experience_list[ed[0] - 1]:
                designation_organization = (experience_list[ed[0] - 1]).split(' at ')
                experience_details_dict['Designation'] = designation_organization[0]
                experience_details_dict['Organization'] = designation_organization[1]
            else:
                experience_details_dict['Designation/Organization'] = experience_list[ed[0] - 1]
                
            experience_details_dict['Start_Date'] = ed[1]
            experience_details_dict['End_Date'] = ed[2]
            experience_details_dict['Num_Years'] = ed[3]
            experience_details_dict['Num_Months'] = ed[4]
            
            if i + 1 < len(experiences_found):
                if experiences_found[i][0] + 1 < experiences_found[i + 1][0] - 1:
                    experience_description = ''
                    for exp_desc in experience_list[experiences_found[i][0] + 1:experiences_found[i + 1][0] - 1]:
                        experience_description += exp_desc
                    experience_details_dict['Description'] = experience_description
                else:
                    experience_details_dict['Description'] = None
            else:
                if experience_list[experiences_found[i][0] + 1:]:
                    experience_description = ''
                    for exp_desc in experience_list[experiences_found[i][0] + 1:]:
                        experience_description += exp_desc
                    experience_details_dict['Description'] = experience_description
                else:
                    experience_details_dict['Description'] = None
                    
            
            experience_details.append(experience_details_dict)
            
        return experience_details
        
    except Exception as e:
        # print('Error:', e)
        return None

    
# Returns the Total Experience in Months
def get_total_experience(experience_list):
    
    if experience_list is None:
        return 0
    
    # Getting Min Start_Date & Max End_Date
    min_start_date = datetime.max
    max_end_date = datetime.min
    for exp_dict in experience_list:
        if 'Start_Date' in exp_dict:
            start_date = datetime.strptime(exp_dict['Start_Date'], '%B %Y')
            if start_date < min_start_date:
                min_start_date = start_date
        if 'End_Date' in exp_dict:
            if exp_dict['End_Date'] == present_string:
                today = date.today()
                max_end_date = datetime(today.year, today.month, today.day)
            else:
                end_date = datetime.strptime(exp_dict['End_Date'], '%B %Y')
                if end_date > max_end_date:
                    max_end_date = end_date
                
    total_experience_num_months = ((max_end_date.year - min_start_date.year) * 12) + (max_end_date.month - min_start_date.month)
    return total_experience_num_months


# Extracts seperate fields from the text of a CV
def extract_fields_from_cv(cv):

    cv_fields = dict()
    
    # Setting 'Personal' Info.
    cv_personal_info = parse_personal_info(cv)
    cv_fields['Name'] = cv_personal_info[0]
    cv_fields['Location'] = cv_personal_info[1]
    cv_fields['Domain'] = cv_personal_info[2]
    
    # For finding Indices
    summary_idx = experience_idx = education_idx = -1
    
    # Finding Indices of respective fields
    try:
        summary_idx = cv.index('Summary')
        
    except Exception as e:
        cv_fields['Summary'] = None
        # print('Warning:', e, 'for Person:', cv_fields['Name'])
        
    try:
        experience_idx = cv.index('Experience')
        
        if summary_idx != -1:
            cv_fields['Summary'] = parse_summary(cv[summary_idx:experience_idx])
        
    except Exception as e:
        
        if summary_idx != -1:
            cv_fields['Summary'] = parse_summary(cv[summary_idx:experience_idx])
        
        cv_fields['Experience'] = None
        # print('Warning:', e, 'for Person:', cv_fields['Name'])
        
    try:
        education_idx = cv.index('Education')
        
         # Setting 'Experience' Info.
        if experience_idx != -1:
            cv_fields['Experience'] = parse_experience(cv[experience_idx:education_idx])
        
        # Setting 'Education' Info.
        cv_fields['Education'] = parse_education(cv[education_idx:])
        
    except Exception as e:
        
        # Setting 'Experience' Info.
        if experience_idx != -1:
            cv_fields['Experience'] = parse_experience(cv[experience_idx:])
        
        cv_fields['Education'] = None
        # print('Warning:', e, 'for Person:', cv_fields['Name'])
        
        
    cv_fields['Total Experience (months)'] = get_total_experience(cv_fields['Experience'])
    
    return cv_fields



# Writes the cv_details dictionary into a Dataframe
def cv_dict_to_list(cv_details):
    
    # Adding 'Personal' Info
    cv_details_list = [cv_details['Name'], cv_details['Location'], cv_details['Domain']]
    
    # Adding 'Summary' Info
    if 'Summary' in cv_details:
        cv_details_list.append(cv_details['Summary'])
    else:
        cv_details_list.append(None)
    
    # Adding 'Experience' Info
    if 'Experience' in cv_details:
        cv_details_list.append(cv_details['Experience'])
    else:
        cv_details_list.append(None)
        
    # Adding 'Total Experience (months)'
    if 'Total Experience (months)' in cv_details:
        cv_details_list.append(cv_details['Total Experience (months)'])
    else:
        cv_details_list.append(0)
    
    # Adding 'Education' Info
    if 'Education' in cv_details:
        cv_details_list.append(cv_details['Education'])
    else:
        cv_details_list.append(None)
        
    return cv_details_list
    

# Completely Processes a single PDF
def process_pdf(pdf_file_path, s3_link):
    
    cv_df = pd.DataFrame(columns = linkedin_cvs_col_names)

    try:
        # Extracting Raw Text from PDF file
        raw_text = extract_text_from_pdf_file(pdf_file_path)

        # Cleaning the Raw Text
        cleaned_text = clean_text(raw_text)

        # Splitting individual CVs from PDF conatining multiple CVs 
        cvs = split_cvs(cleaned_text)

        # Extracting Fields from CVs
        for i, cv in enumerate(cvs):

            # Extracting Fields
            cv_detail = extract_fields_from_cv(cv)

            # Appending to Dataframe
            cv_details_list = cv_dict_to_list(cv_detail)
            cv_df.loc[i + 1] = cv_details_list + [s3_link]

        return cv_df, True
            
    except Exception as e:
        # print('Error while Parsing:', os.path.basename(pdf_file_path))
        return cv_df, False



# ### Processing Single PDF

# # Filename of the CV (PDF)
# pdf_filename = '11.pdf'

# # Path to the CV (PDF) file
# pdf_file_path = os.path.join(testing_cvs_dir, pdf_filename)


# st = time.time()

# cv_df, _ = process_pdf(pdf_file_path)

# et = time.time()

# print('Time Taken:', round(et - st, 2))
# cv_df



# Parses all the PDF files in the given directory
def bulk_parse_pdfs(dir_path, cv_s3_links, multithreading=True):

    # Getting all PDF Files
    pdf_files = [os.path.join(dir_path, pdf_file) for pdf_file in sorted(os.listdir(dir_path)) if pdf_file.lower().endswith('.pdf')]

    # CVs S3 Links
    s3_links = [cv_s3_link for pdf_file in pdf_files for cv_s3_link in cv_s3_links if pdf_file.split('/')[-1] in cv_s3_link]
    
    if multithreading:
        
        # Text Extraction from PDFs
        cvs_dfs = Parallel(n_jobs=-1, backend='multiprocessing')(map(delayed(process_pdf), pdf_files, s3_links))

        # Gathering Results
        cvs_df = pd.concat([cv_df for cv_df, _ in cvs_dfs], ignore_index=True)
        accuracy = round(sum([1 for _, response in cvs_dfs if response]) / len(pdf_files), 2)

        # Removing Negative Experience CVs
        cvs_df = cvs_df[cvs_df['Total Experience (months)'] >= 0]

        return cvs_df, accuracy

    else:

        # For all Files in the Directory
        accuracy = 0
        cvs_df = pd.DataFrame(columns = linkedin_cvs_col_names)
        cvs_dict, pdfs_dict, s3_dict = dict(), dict(), dict()
        for pdf_file, s3_link in zip(pdf_files, s3_links):
                
            # print('Processing:', pdf_file)

            # Processing the PDF file
            cv_df, completely_processed = process_pdf(pdf_file, s3_link)
            
            # For Accuracy
            if completely_processed:
                accuracy += 1

            cvs_df = cvs_df.append(cv_df, ignore_index=True)
        
        # Removing Negative Experience CVs
        cvs_df = cvs_df[cvs_df['Total Experience (months)'] >= 0]
        
        # Calculating Accuracy
        accuracy = round(accuracy / len(os.listdir(dir_path)), 2)
        
        return cvs_df, accuracy
