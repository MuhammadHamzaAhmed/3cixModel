import os

### Constants used within the Project


# Delete all Old Files (For Debugging)
delete_all = True

# Main CVs Directory
cvs_dir = './CVs'

# Directory containing the Testing CVs in PDF Format
testing_cvs_dir = os.path.join(cvs_dir, 'Testing')

# Directory for Downloading CVs from S3
downloaded_cvs_dir = os.path.join(cvs_dir, 'Others')

# Directory for Downloading LinkedIn CVs from S3
downloaded_linkedin_cvs_dir = os.path.join(cvs_dir, 'LinkedIn')

# Directory for Results
results_dir = './results'
results_filename = 'result'

# Directory for Requests
requests_dir = './requests'
requests_filename = 'request'

# Directory for Logs
logs_dir = './logs'
logs_filename = 'log'

# Removing NaNs with this variable in Results Dictionary
nans_replace_str = 'Unavailable'


# Filename for storing extracted CVs
cvs_filename = './CVs'

# Filename for JDs
jds_filename = 'JDs.csv'

# PDF Page Separator
pdf_page_separator = '**********'

# String for recognizing Next CV
next_cv_string = '\n\n\x0c'

# String for recognizing LinkedIn Recommendations
linkedin_recommendation_string = 'people have recommended'

# String for 'Present' Experience
present_string = 'Present'

# Columns for CVs Dataframes
nans_replace_str = 'Unavailable'
cvs_col_names = ['ID', 'Name', 'Email', 'S3 Path', 'Total Experience (months)', 'Location', 'Domain']
linkedin_cvs_col_names = ['Name', 'Location', 'Domain', 'Summary', 'Experience', 'Total Experience (months)', 'Education', 'S3 Path']

# Paths for Object Detection Model Files
model_files_dir = './model'
model_file = os.path.join(model_files_dir, 'model_best.pth')
model_labels_file = os.path.join(model_files_dir, 'Labels.txt')
model_config_file = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'

# Paths for CV Classification Model Files
use_clf_model = True
clf_model_file = os.path.join(model_files_dir, 'cv_classification.model')
cv_labels = {0:'Others',  1:'LinkedIn'}
downloaded_clf_cvs_dir = os.path.join(cvs_dir, 'Classification')

# AWS Keys
ACCESS_KEY = 'AKIAX6OFYQYFSUTYRJGH'
SECRET_KEY = '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'
BUCKET_NAME = '3cix'
S3_UPLOAD_PREFIX = 'results'