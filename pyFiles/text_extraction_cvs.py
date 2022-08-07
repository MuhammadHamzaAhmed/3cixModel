# Basics Libraries
import io
import re

from joblib import Parallel, delayed
from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
# For PDF Text Extraction
from pdfminer3.pdfpage import PDFPage
from pyFiles.Ranker import Converter

# Importing Constants
from pyFiles.constants import *


# For Data Handling


### Text Extraction


# Cleans the Raw Text
def clean_text(text):
    # Removing Spaces
    text = ' '.join(text.split())

    # Removing Unwanted Characters
    unwanted_chars = ['"', "'", '‘', '’', '~', '|', '\n', '\xa0', '\x0c']
    for uc in unwanted_chars:
        text = text.replace(uc, ' ')

    # Converting to lower case
    text = text.lower()
    return text


# Extracts email from CV Text
def extract_email(text):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email = re.findall(regex, text)
    if email:
        return email[0]
    else:
        return


# Extracts Email & Text from PDF file
def extract_pdf_details(pdf_file_path, s3_link):
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

        # Cleaning Text
        text = clean_text(text)

        # Getting Email
        email = extract_email(text)

        # PDF Filename
        pdf_filename = pdf_file_path

        return email, pdf_filename, s3_link, text
    except Exception as e:
        print('Error while Processing:', pdf_file_path)
        print('Error:', e)


# Extracts Text from all the PDF files in the given directory
def bulk_parse_pdfs(dir_path, cv_s3_links, multithreading=True):
    # Getting all PDF Files
    # _ = Converter(dir_path, "CVs/Copy")
    # dir_path = "CVs/Copy"
    pdf_files = [os.path.join(dir_path, pdf_file) for pdf_file in sorted(os.listdir(dir_path)) if
                 pdf_file.lower().endswith('.pdf')]
    # CVs S3 Links
    s3_links = [cv_s3_link for pdf_file in pdf_files for cv_s3_link in cv_s3_links if
                pdf_file.split('/')[-1] in cv_s3_link]
    if multithreading:
        # Text Extraction from PDFs
        cvs_details = Parallel(n_jobs=-1, backend='multiprocessing')(
            map(delayed(extract_pdf_details), pdf_files, s3_links))

        # Converting to Dictionary
        emails_dict, cvs_dict, pdfs_dict, s3_dict = dict(), dict(), dict(), dict()
        for i, cvs_detail in enumerate(cvs_details):
            email, pdf_filename, s3_link, text = cvs_detail
            emails_dict[i] = email
            cvs_dict[i] = text
            s3_dict[i] = s3_link
            pdfs_dict[i] = pdf_filename
        return emails_dict, cvs_dict, pdfs_dict, s3_dict

    else:
        # For all Files in the Directory
        emails_dict, cvs_dict, pdfs_dict, s3_dict = dict(), dict(), dict(), dict()
        for i in range(len(pdf_files)):
            pdf_file, s3_link = pdf_files[i], s3_links[i]

            # Text Extraction from PDF
            cv_email, cv_pdf_filename, cv_s3_link, cv_text = extract_pdf_details(pdf_file, s3_link)

            emails_dict[i] = cv_email
            cvs_dict[i] = cv_text
            pdfs_dict[i] = cv_pdf_filename
            s3_dict[i] = cv_s3_link

        return emails_dict, cvs_dict, pdfs_dict, s3_dict
