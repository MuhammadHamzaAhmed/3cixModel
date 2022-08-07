# Data Handling
import os.path
import traceback

# NLTK
import nltk

from pyFiles import ranking_cvs as cvs_ranking, ranking_cvs_linkedin as linkedin_ranking, \
    text_extraction_cvs as cvs_text, text_extraction_cvs_linkedin as linkedin_text
from pyFiles.Ranker import *
from pyFiles.utilities import *


# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

from flask import Flask, request
from flask_restful import Resource, Api


def process_cvs(jd, cv_links, request_id, logger):
    try:
        start_time = time.time()
        request_cvs_dir = os.path.join(downloaded_cvs_dir, request_id)
        print('Downloading CVs..')
        s = time.time()
        delete_copy()
        download_files(cv_links, request_cvs_dir, multithreading=False)
        e = time.time()
        print('Time Taken in Downloading CVs: ' + str(round(e - s, 2)) + ' secs\n')

        print('Extracting Text..')
        s = time.time()
        emails_dict, cvs_dict, pdfs_dict, s3_dict = cvs_text.bulk_parse_pdfs(request_cvs_dir, cv_links,
                                                                             multithreading=False)
        e = time.time()
        print('Time Taken in Text Extraction: ' + str(round(e - s, 2)) + ' secs\n')

        print('Getting Model Predictions..')
        s = time.time()
        print("Extracting Experience..")
        cvs_df = cvs_ranking.get_cv_details_df(emails_dict, pdfs_dict, s3_dict)
        cvs_df.to_csv("Test.csv")
        print("Prediction:", cvs_df)
        e = time.time()
        # changing done by me
        only_files = [f for f in listdir(request_cvs_dir) if isfile(join("", f)) and f[-3:] == "pdf"]

        check_location = True if jd['Location'] is not None else False
        check_job_title = True if jd['Job Title'] is not None else False
        check_required_experience_months = True if jd['Required Experience'] is not None else False

        print('Ranking CVs..')
        s = time.time()
        ranked_cvs = cvs_ranking.rank_cvs(jd, cvs_dict, cvs_df, check_location, check_job_title,
                                          check_required_experience_months)
        e = time.time()
        for i, process in enumerate(only_files):
            loc = None
            if check_location:
                loc = jd['Location']
            requiredSkills = jd['Required Skills']
            parser = Parser(process, requiredSkills, requiredLocation=loc)
            parser.read()
            parser.extractName()
            parser.extractSkills()
            parser.rankResume()
            ranked_cvs.Name.iloc[i] = parser.name
            ranked_cvs.Domain.iloc[i] = parser.skills
            ranked_cvs['Similarity Score'].iloc[i] = parser.ranked
        print('Time Taken in Model Predictions: ' + str(round(e - s, 2)) + ' secs\n')
        print('Time Taken in Ranking: ' + str(round(e - s, 2)) + ' secs\n')

        end_time = time.time()
        print('Total Time Taken to Process all CVs: ' + str(round(end_time - start_time, 2)) + ' secs')
        print(
            'Average Time Taken to Process a CV: ' + str(round((end_time - start_time) / len(cv_links), 2)) + ' secs\n')
        return ranked_cvs
    except Exception as e:
        print('\nAn Error occured while Processing Other CVs')
        print('Error Details:\n' + str(e) + '\n')
        traceback.print_exc()


def process_linkedin_cvs(jd, linkedin_cv_links, request_id):
    try:
        start_time = time.time()
        request_linkedin_cvs_dir = os.path.join(downloaded_linkedin_cvs_dir, request_id)
        print('Downloading LinkedIn CVs..')
        s = time.time()
        download_files(linkedin_cv_links, request_linkedin_cvs_dir, multithreading=True)
        e = time.time()
        print('Time Taken in Downloading LinkedIn CVs: ' + str(round(e - s, 2)) + ' secs\n')

        print('Extracting LinkedIn Text..')
        s = time.time()
        cvs_df, accuracy = linkedin_text.bulk_parse_pdfs(request_linkedin_cvs_dir, linkedin_cv_links,
                                                         multithreading=True)
        e = time.time()
        print('Time Taken in LinkedIn Text Extraction: ' + str(round(e - s, 2)) + ' secs\n')

        check_location = True if jd['Location'] is not None else False
        check_job_title = True if jd['Job Title'] is not None else False
        check_required_experience_months = True if jd['Required Experience'] is not None else False

        print('Ranking LinkedIn CVs..')
        s = time.time()
        ranked_linkedin_cvs = linkedin_ranking.rank_cvs(jd, cvs_df, check_location, check_job_title,
                                                        check_required_experience_months)
        e = time.time()
        print('Time Taken in Ranking: ' + str(round(e - s, 2)) + ' secs\n')

        end_time = time.time()
        print('Total Time Taken to Process all LinkedIn CVs: ' + str(round(end_time - start_time, 2)) + ' secs')
        print('Average Time Taken to Process a LinkedIn CV: ' + str(
            round((end_time - start_time) / len(linkedin_cv_links), 2)) + ' secs\n')

        return ranked_linkedin_cvs

    except Exception as e:
        print('\nAn Error occured while Processing LinkedIn CVs')
        print('Error Details:\n' + str(e) + '\n')
        traceback.print_exc()


def combine_cvs_results(ranked_cvs, ranked_linkedin_cvs, top_n_cvs, logger):
    try:
        combined_rankings = pd.concat([ranked_cvs, ranked_linkedin_cvs], axis=0, ignore_index=True)
        combined_rankings = combined_rankings.sort_values('Similarity Score', ascending=False)
        combined_rankings = combined_rankings.sort_values(['Similarity Score', 'Total Experience (months)'],
                                                          ascending=[False, False])
        combined_rankings.reset_index(drop=True, inplace=True)
        return combined_rankings
    except Exception as e:
        print('\nAn Error occured while Combining Results')
        print('Error Details:\n' + str(e) + '\n')
        traceback.print_exc()


# Removes Unneeded Columns
def remove_cols(df):
    if 'Experience Difference' in df.keys():
        df.pop('Experience Difference')
    df.pop('Similarity Score')
    return df


def get_top_n(cvs_df, top_n_cvs):
    return cvs_df.iloc[:top_n_cvs, :]


def df_to_dict(cvs_df):
    cvs_dict = cvs_df.to_dict(orient='index')
    cvs_dict = [cvs_dict[k] for k in cvs_dict.keys()]
    return cvs_dict


def process_all_cvs(jd, request_id, result):
    try:
        print('\nStarted Processing - Request:', request_id, '\n')
        s = time.time()
        current_datetime = get_current_datetime()
        logger = create_logger(request_id, current_datetime)
        print('Started Processing - Request: ' + request_id + '\n')
        write_to_json(requests_dir, requests_filename, request_id, current_datetime, jd)  # 1
        print("Writing to JSON Done")
        results_api_endpoint = ""
        top_n_cvs = int(jd['Number of CVs']) if 'Number of CVs' in jd.keys() else 10
        #
        # Getting CVs Links
        cv_links = jd['CV Links']
        linkedin_cv_links = jd['LinkedIn CV Links']
        #
        # # Checking CV Links
        cv_links_obtained = not is_nan(cv_links)
        linkedin_cv_links_obtained = not is_nan(linkedin_cv_links)
        print("CV LINKS: ", cv_links_obtained)
        print("Linked in CV LINKS: ", linkedin_cv_links_obtained)
        #
        print('CV Links Obtained: ' + str(cv_links_obtained) + ', Count: ' + str(get_length(cv_links)))
        print('LinkedIn CV Links Obtained: ' + str(linkedin_cv_links_obtained) + ', Count: ' + str(
            get_length(linkedin_cv_links)) + '\n')
        #
        # # Threads for Processing CVs
        cvs_thread, linkedin_cvs_thread = None, None
        if cv_links_obtained:
            cvs_thread = ReturnThread(target=process_cvs, args=(jd, cv_links, request_id, logger))
            cvs_thread.start()
        if linkedin_cv_links_obtained:
            linkedin_cvs_thread = ReturnThread(target=process_linkedin_cvs,
                                               args=(jd, linkedin_cv_links, request_id, logger))
            linkedin_cvs_thread.start()

        # Getting CVs Processing Results
        ranked_cvs, ranked_linkedin_cvs = None, None
        if cvs_thread:
            cvs_thread.join()
        if linkedin_cvs_thread:
            linkedin_cvs_thread.join()

        # Checking if Results obtained
        cvs_obtained, linkedin_cvs_obtained = check_results_obtained(ranked_cvs, ranked_linkedin_cvs)
        print(ranked_cvs)
        print('CV Rankings Obtained: ' + str(cvs_obtained) + ', Count: ' + str(get_length(ranked_cvs)))
        print('LinkedIn CV Rankings Obtained: ' + str(linkedin_cvs_obtained) + ', Count: ' + str(
            get_length(ranked_linkedin_cvs)) + '\n')

        # Combining CVs Results
        ranked_cvs_dict = {}

        if cvs_obtained and linkedin_cvs_obtained:
            ranked_cvs_df = combine_cvs_results(ranked_cvs, ranked_linkedin_cvs, top_n_cvs, logger)
            ranked_cvs_df = remove_cols(ranked_cvs_df)
            top_n_ranked_cvs_df = get_top_n(ranked_cvs_df, top_n_cvs)
            ranked_cvs_dict = df_to_dict(top_n_ranked_cvs_df)

        elif cvs_obtained:
            ranked_cvs_df = remove_cols(ranked_cvs)
            top_n_ranked_cvs_df = get_top_n(ranked_cvs_df, top_n_cvs)
            ranked_cvs_dict = df_to_dict(top_n_ranked_cvs_df)
            print('LinkedIn CVs Not Provided\n')

        elif linkedin_cvs_obtained:
            ranked_cvs_df = remove_cols(ranked_linkedin_cvs)
            top_n_ranked_cvs_df = get_top_n(ranked_cvs_df, top_n_cvs)
            ranked_cvs_dict = df_to_dict(top_n_ranked_cvs_df)
            print('Other CVs Not Provided\n')

        else:
            print('CVs Not Provided\n')
            ranked_cvs_dict = {'data': 'CVs Not Provided'}

        ranked_cvs_dict = remove_nans_from_dict(ranked_cvs_dict, nans_replace_str)

        write_to_json(results_dir, results_filename, request_id, current_datetime, ranked_cvs_dict)
        ranked_cvs_dict = {'data': json.dumps(ranked_cvs_dict)}

        api_status, api_response = post_results(results_api_endpoint, ranked_cvs_dict)
        if api_status:
            print('API Response: True\n')
        else:
            print('API Response: False, ' + str(api_response) + '\n')

        e = time.time()
        print('\nFinished Processing - Request:', request_id)
        print('Time Taken in Processing:', round(e - s, 2), ' secs\n')

        result.append(ranked_cvs_dict)
        results.append(api_response)
    except Exception as e:
        print('\nAn Error occured while Processing All CVs')
        print('Error Details:\n' + str(e) + '\n')
        traceback.print_exc()


### Flask API

# Initializing Flask App
app = Flask(__name__)
api = Api(app)

# Disabling Logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# # API Arguments
# api_args = reqparse.RequestParser()
# api_args.add_argument('Callback Url', type=str, help="'Callback Url' is required.", required=True)
# api_args.add_argument('Number of CVs', type=int, help="'Industry' is required.", required=False)
# api_args.add_argument('Industry', type=str, help="'Industry' is required.", required=False)
# api_args.add_argument('Soft Skills', type=str, help="'Soft Skills' is required.", required=False)
# api_args.add_argument('Location', type=str, help="'Location' is required.", required=False)
# api_args.add_argument('Job Title', type=str, help="'Job Title' is required.", required=False)
# api_args.add_argument('Required Experience', type=int, help="'Required Experience' is required.", required=False)
# api_args.add_argument('Tools Handling Experience', type=str, help="'Tools Handling Experience' is required.",
#                       required=False)
# api_args.add_argument('Qualification', type=str, help="'Tools Handling Experience' is required.", required=False)
# api_args.add_argument('Mandatory Requirement', type=str, help="'Mandatory Requirement' is required.", required=False)
# api_args.add_argument('Certification', type=str, help="'Certification' is required.", required=False)
# api_args.add_argument('Detailed Job Description', type=str, help="'Detailed Job Description' is required.",
#                       required=False)
# api_args.add_argument('CV Links', action='append', help="'CV Links' are required.", required=False)
# api_args.add_argument('LinkedIn CV Links', action='append', help="'CV Links' are required.", required=False)


# class Rank(Resource):
#     # POST API
#     def post(self):
#         # Getting Arguments
#         args = api_args.parse_args()
#         jd = dict(args)
#         request_id = str(jd['Callback Url'].split('/')[-1])
#
#         # Creating Directory
#         create_request_cvs_dir(request_id)
#
#         # Deleting old files
#         delete_files(request_id)
#
#         # Starting Processing Thread Thread
#         worker = Thread(target=process_all_cvs, args=(jd, request_id))
#         worker.setDaemon(True)
#         worker.start()
#
#         # results_path = os.path.join(BUCKET_NAME, S3_UPLOAD_PREFIX, results_filename)
#
#         return {'results_link': jd['Callback Url']}


# Adding API Endpoint
# api.add_resource(Rank, '/Resume_extraction')
def delete_copy():
    lis = [os.path.join("CVs/Copy", pdf) for pdf in os.listdir("CVs/Copy")]
    for dir in lis:
        os.remove(dir)

@app.route('/Resume_extraction', methods=['POST'])
def resume_seg():
    try:
        request_data = request.get_json()
        urls = request_data['Urls'].split(",")
        skills = request_data['Skills']
        country_desired = request_data['DesiredCountries']
        required_skills_list = skills.lower().split(",")

        jd = {
            "CV Links": urls,
            "Number of CVs": len(urls),
            "Job Title": "",
            "Location": country_desired,
            "Required Skills": required_skills_list,
            "LinkedIn CV Links": []
        }
        create_request_cvs_dir("9d983d07-84dd-4bd3-a861-6dbffe0b041f")
        delete_files("9d983d07-84dd-4bd3-a861-6dbffe0b041f")
        delete_copy()
        results = []
        worker = Thread(target=process_all_cvs, args=(jd, "9d983d07-84dd-4bd3-a861-6dbffe0b041f", results))
        worker.setDaemon(True)
        worker.start()
        worker = worker.join()
        df = results[0]
        return df
    except Exception as e:
        print(e)


# Healthcheck
@app.route('/healthcheck')
def healthcheck():
    return 'healthy', 200


if __name__ == '__main__':

    # if delete_all:
    #     delete_all_old_files()

    create_dirs()
    app.run(host='0.0.0.0', port=3000, debug=True, threaded=True)


"""
{"Skills":"openstack,perl,mysql,router,container tools,load balancing (computing),fault tolerance,platform as a service,semantic html,javascript,virtualization platform",
"Urls":"https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2F1ddc0de1-a325-4ab2-9dc0-bb87ad27a78f.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2F5369f127-1612-4c29-af43-d51c61328722.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2F6cc768ef-ac00-44f5-a56b-2dd77f721bc0.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2Fab703494-6134-4e74-9aeb-49dfdf1bd8bc.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2F3045f682-2afc-4bd7-abe0-7f4d908c3f93.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2F18d38e28-c3dc-476a-b35d-66aea1a7719e.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2Fb11ebbd9-75dd-4c38-836a-b4f464279e58.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2Fb8db6a96-87b6-4412-be7e-f49f6f1111ee.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2Fac74d84f-66a0-4258-8e8b-196acca16c03.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2Ffe9c531a-82dd-4c36-a1b5-922424531c14.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2F082b9d9a-3de5-48d8-9425-33eee6dd51ca.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2Ff53d750a-cacc-4a31-ba55-27dc47870832.pdf,https://3cixstorage.blob.core.windows.net/resumes/resumes%2Fd1190d12-1054-4af1-8759-bbe5917071fe%2Fba10c035-61b3-4a52-be42-9fa824ec58b7.pdf",
"DesiredCountries":"Pakistani"}
"""