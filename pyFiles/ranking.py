# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.5 (default, Sep  4 2020, 07:30:14) 
# [GCC 7.3.0]
# Embedded file name: /home/uzair/Desktop/nlp-api/ranking_cvs.py
# Compiled at: 2021-10-17 06:20:11
# Size of source mod 2**32: 23560 bytes
import os, re, random, numpy as np, pandas as pd
from collections import Counter
from joblib import Parallel, delayed
import cv2, PIL, pytesseract
from pdfminer3.pdfpage import PDFPage
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator, TextConverter
from pdf2image import convert_from_path
from datetime import date, datetime
from dateparser.search import search_dates
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (50, 50)
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz
from constants import *

def clean_text(raw_text):
    text = ' '.join(raw_text.split())
    unwanted_chars = [
     '"', "'", '‘', '’', '~', '|', '\n', '\xa0', '\x0c']
    for uc in unwanted_chars:
        text = text.replace(uc, ' ')

    text = ' '.join(text.split())
    text = text.lower()
    return text


def extract_text_from_image(img):
    try:
        text = pytesseract.image_to_string(img)
        text = clean_text(text)
        return text
    except Exception as e:
        print('Error while Extracting Text from OCR:')
        print('Error:', e)


def pdf_to_imgs(pdf_file_path, img_size=(900, 1200)):
    imgs = convert_from_path(pdf_file_path)
    images = [np.array(img.resize(img_size, PIL.Image.BICUBIC)) for img in imgs]
    return images


def extract_email(text):
    regex = '\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
    email = re.findall(regex, text)
    if email:
        return email[0]
    else:
        return


def get_labels(labels_file_path):
    f = open(labels_file_path, 'r')
    labels = f.read().split()
    f.close()
    return labels


def get_detectron_predictor(labels):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config_file))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = model_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)
    predictor = DefaultPredictor(cfg)
    return predictor


def get_model_predictions(images, predictor):
    all_preds = dict()
    for i, img in enumerate(images):
        outputs = predictor(img)
        pred_boxes = outputs['instances'].pred_boxes.to('cpu').tensor.numpy().astype(int)
        pred_classes = outputs['instances'].pred_classes.to('cpu').numpy()
        pred_scores = outputs['instances'].scores.to('cpu').numpy()
        page_preds = dict()
        page_preds['bboxes'] = pred_boxes
        page_preds['classes'] = pred_classes
        page_preds['scores'] = pred_scores
        all_preds[i + 1] = page_preds

    return all_preds


def generate_colors(n):
    rgb_values = []
    hex_values = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for _ in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        rgb_values.append((r, g, b))

    return rgb_values


def visualize_model_predictions(images, labels, model_predictions, conf_thresh=0.8):
    imgs = []
    cls_colors = generate_colors(len(labels))
    for k, image in enumerate(images):
        curr_img_preds = model_predictions[(k + 1)]
        img = image.copy()
        for pred_box, pred_cls, pred_score in zip(curr_img_preds['bboxes'], curr_img_preds['classes'], curr_img_preds['scores']):
            if pred_score >= conf_thresh:
                pt = (
                 pred_box[0] - 5, pred_box[1])
                pt1 = tuple(pred_box[:2])
                pt2 = tuple(pred_box[2:])
                cls = labels[pred_cls]
                color = cls_colors[pred_cls]
                conf = round(pred_score, 3)
                img = cv2.rectangle(img, pt1, pt2, color, 1)
                img = cv2.putText(img, cls + ' ' + str(conf), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        imgs.append(img)

    for i, img in enumerate(imgs):
        (plt.subplot(len(imgs), 1, i + 1), plt.imshow(img, cmap='gray'))
        plt.title('Page ' + str(i + 1))

    plt.show()


def get_dates_from_text(text):
    start_date, end_date = (0, 0)
    searched_dates = search_dates(text)
    unneeded_dates_strs = 'today present still on in to from between then before after start end next new day % +'.split()
    required_dates = []
    for dt in searched_dates:
        unneeded_dates_str_present = False
        for uds in unneeded_dates_strs:
            if uds in dt[0]:
                unneeded_dates_str_present = True

        if not unneeded_dates_str_present:
            if dt[1] < datetime.today() and dt[1].year > 1950:
                required_dates.append(dt)

    present_len = 15
    present_strs = ['present', 'still', 'today']
    present_date = False
    for dt in required_dates:
        dt_text_start_idx = text.index(dt[0])
        dt_text_end_idx = dt_text_start_idx + len(dt[0])
        if dt_text_end_idx < len(text):
            dt_text_next_str = text[dt_text_end_idx:dt_text_end_idx + present_len]
        else:
            dt_text_next_str = text[dt_text_end_idx:]
        for ps in present_strs:
            if ps in dt_text_next_str:
                present_date = True

    dates = [dt[1] for dt in required_dates]
    if not dates:
        return
    else:
        start_date = min(dates)
        if present_date:
            end_date = date.today()
        else:
            end_date = max(dates)
        return (start_date, end_date)


def get_experience(images, model_predictions, labels, conf_thresh=0.9):
    exp_idx = labels.index('Experience')
    exp_found = False
    for k in model_predictions:
        if exp_idx in model_predictions[k]['classes']:
            exp_found = True

    if not exp_found:
        return
    else:
        imgs = []
        for k, image in enumerate(images):
            curr_img_preds = model_predictions[(k + 1)]
            img = image.copy()
            for pred_box, pred_cls, pred_score in zip(curr_img_preds['bboxes'], curr_img_preds['classes'], curr_img_preds['scores']):
                if pred_cls == exp_idx and pred_score >= conf_thresh:
                    img_chunk = image[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2], :]
                    imgs.append(img_chunk)

        text = ''
        for img in imgs:
            text += pytesseract.image_to_string(img)

        text = clean_text(text)
        dates_result = get_dates_from_text(text)
        total_experience_num_months = 0
        if dates_result:
            start_date, end_date = dates_result
            total_experience_num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        return total_experience_num_months


def get_name(images, model_predictions, labels, conf_thresh=0.5):
    pd_idx = labels.index('Name')
    pd_found = False
    for k in model_predictions:
        if pd_idx in model_predictions[k]['classes']:
            pd_found = True

    if not pd_found:
        return
    else:
        imgs, conf_scores = [], []
        for k, image in enumerate(images):
            curr_img_preds = model_predictions[(k + 1)]
            img = image.copy()
            for pred_box, pred_cls, pred_score in zip(curr_img_preds['bboxes'], curr_img_preds['classes'], curr_img_preds['scores']):
                if pred_cls == pd_idx and pred_score >= conf_thresh:
                    img_chunk = image[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2], :]
                    imgs.append(img_chunk)
                    conf_scores.append(pred_score)

        img = imgs[np.argmax(conf_scores)]
        text = pytesseract.image_to_string(img)
        name = clean_text(text)
        return name


def get_location(images, model_predictions, labels, conf_thresh=0.5):
    loc_idx = labels.index('Location')
    loc_found = False
    for k in model_predictions:
        if loc_idx in model_predictions[k]['classes']:
            loc_found = True

    if not loc_found:
        return
    else:
        imgs, conf_scores = [], []
        for k, image in enumerate(images):
            curr_img_preds = model_predictions[(k + 1)]
            img = image.copy()
            for pred_box, pred_cls, pred_score in zip(curr_img_preds['bboxes'], curr_img_preds['classes'], curr_img_preds['scores']):
                if pred_cls == loc_idx and pred_score >= conf_thresh:
                    img_chunk = image[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2], :]
                    imgs.append(img_chunk)
                    conf_scores.append(pred_score)

        img = imgs[np.argmax(conf_scores)]
        text = pytesseract.image_to_string(img)
        location = clean_text(text)
        return location


def get_professional_domain(images, model_predictions, labels, conf_thresh=0.5):
    pd_idx = labels.index('Professional_Domain')
    pd_found = False
    for k in model_predictions:
        if pd_idx in model_predictions[k]['classes']:
            pd_found = True

    if not pd_found:
        return
    else:
        imgs, conf_scores = [], []
        for k, image in enumerate(images):
            curr_img_preds = model_predictions[(k + 1)]
            img = image.copy()
            for pred_box, pred_cls, pred_score in zip(curr_img_preds['bboxes'], curr_img_preds['classes'], curr_img_preds['scores']):
                if pred_cls == pd_idx and pred_score >= conf_thresh:
                    img_chunk = image[pred_box[1]:pred_box[3], pred_box[0]:pred_box[2], :]
                    imgs.append(img_chunk)
                    conf_scores.append(pred_score)

        img = imgs[np.argmax(conf_scores)]
        text = pytesseract.image_to_string(img)
        professional_domain = clean_text(text)
        return professional_domain


def get_experience_months(experience_string, range_num_months=12):
    if type(experience_string) is str:
        if '-' in experience_string:
            num_years, year_str = experience_string.split()
            years_split = num_years.split('-')
            min_years, max_years = num_years.split('-')
            return int((int(min_years) + int(max_years)) / 2 * range_num_months)
        else:
            num_years, year_str = experience_string.split()
            return int(num_years) * 12
    else:
        if type(experience_string) is int:
            return experience_string * 12
        print("Unexpected Type for 'Required experience'")


def get_string_matching(str1, str2, partial=True):
    if type(str1) is str and type(str2) is str and str1 and str2:
        if partial:
            matching_percentage = round(fuzz.partial_ratio(str1, str2))
        else:
            matching_percentage = round(fuzz.ratio(str1, str2))
        return matching_percentage
    else:
        print('Unexpected Variable type for string Matching')
        return


def get_model_results(predictor, labels, id, email, pdf_file_path, s3_path):
    images = pdf_to_imgs(pdf_file_path, img_size=(900, 1200))
    model_predictions = get_model_predictions(images, predictor)
    experience = get_experience(images, model_predictions, labels, 0.5)
    name = get_name(images, model_predictions, labels, 0.5)
    location = get_location(images, model_predictions, labels, 0.5)
    professional_domain = get_professional_domain(images, model_predictions, labels, 0.5)
    return [
     id, name, email, s3_path, experience, location, professional_domain]


def get_cv_details_df(emails_dict, pdfs_dict, s3_dict, multithreading=False):
    labels = get_labels(model_labels_file)
    predictor = get_detectron_predictor(labels)
    if multithreading:
        predictor_ = [predictor for _ in range(len(pdfs_dict))]
        labels_ = [labels for _ in range(len(pdfs_dict))]
        ids_ = [k for k in emails_dict]
        emails_ = [emails_dict[k] for k in emails_dict]
        pdf_file_paths_ = [pdfs_dict[k] for k in pdfs_dict]
        s3_paths_ = [s3_dict[k] for k in pdfs_dict]
        model_results = Parallel(n_jobs=2, backend='multiprocessing')(map(delayed(get_model_results), predictor_, labels_, ids_, emails_, pdf_file_paths_, s3_paths_))
        cvs_df = pd.DataFrame(model_results, columns=cvs_col_names)
        return cvs_df
    else:
        cvs_df = pd.DataFrame(columns=cvs_col_names)
        for i, k in enumerate(pdfs_dict):
            email = emails_dict[k]
            pdf_file_path = pdfs_dict[k]
            s3_path = s3_dict[k]
            cvs_df.loc[i] = get_model_results(predictor, labels, k, email, pdf_file_path, s3_path)

        return cvs_df


def get_cvs_closest_experiences(cvs_df, required_experience_months, experience_range_threshold=12):
    if required_experience_months < 12:
        print('Unexpected Experience Required')
        return
    else:
        cvs_df['Experience Difference'] = np.abs(cvs_df['Total Experience (months)'] - required_experience_months)
        closest_cvs = cvs_df[(cvs_df['Experience Difference'] <= experience_range_threshold)].copy(deep=True)
        closest_cvs.reset_index(drop=True, inplace=True)
        return closest_cvs


def get_closest_location_cvs(cvs_df, location, matching_ratio=80):
    cvs_df['Location Matched'] = cvs_df.apply((lambda cv: get_string_matching(cv['Location'], location)), axis=1) > matching_ratio
    closest_cvs = cvs_df[(cvs_df['Location Matched'] == True)].copy(deep=True)
    closest_cvs.reset_index(drop=True, inplace=True)
    cvs_df.pop('Location Matched')
    return closest_cvs


def get_closest_job_title_cvs(cvs_df, job_title, matching_ratio=80):
    cvs_df['Domain Matched'] = cvs_df.apply((lambda cv: get_string_matching(cv['Domain'], job_title)), axis=1) > matching_ratio
    closest_cvs = cvs_df[(cvs_df['Domain Matched'] == True)].copy(deep=True)
    closest_cvs.reset_index(drop=True, inplace=True)
    cvs_df.pop('Domain Matched')
    return closest_cvs


def preprocess_text(text, remove_stop_words=True, lemmatization=True):
    main_words = re.sub('[^a-zA-Z]', ' ', str(text))
    main_words = main_words.lower().split()
    if remove_stop_words:
        main_words = [w for w in main_words if w not in set(stopwords.words('english'))]
    if lemmatization:
        main_words = [WordNetLemmatizer().lemmatize(w) for w in main_words if len(w) > 1]
    main_words = [w for w in main_words if len(w) > 1]
    return main_words


def get_jd_text(jd):
    jd_text = ''
    jd_text += jd['Industry'] + ' ' if jd['Industry'] is not None else ''
    jd_text += jd['Soft Skills'] + ' ' if jd['Soft Skills'] is not None else ''
    jd_text += jd['Job Title'] + ' ' if jd['Job Title'] is not None else ''
    jd_text += jd['Tools Handling Experience'] + ' ' if jd['Tools Handling Experience'] is not None else ''
    jd_text += jd['Mandatory Requirement'] + ' ' if jd['Mandatory Requirement'] is not None else ''
    jd_text += jd['Certification'] + ' ' if jd['Certification'] is not None else ''
    jd_text += jd['Detailed Job Description'] + ' ' if jd['Detailed Job Description'] is not None else ''
    jd_text = jd_text.replace('\xa0', ' ')
    jd_words = preprocess_text(jd_text)
    sorted_jd_words = sorted(jd_words)
    jd_words = Counter(sorted_jd_words)
    return jd_words


def get_cv_text(cv):
    cv_text = ''
    cv_text += cv['Domain'] + ' ' if cv['Domain'] is not None else ''
    cv_text += cv['Summary'] + ' ' if cv['Summary'] is not None else ''
    if type(cv['Experience']) is list and len(cv['Experience']) > 0 and type(cv['Experience']) is list and type(cv['Experience'][0]) is dict:
        cv_exp = ''
        for exp in cv['Experience']:
            if 'Designation' in exp:
                cv_exp += exp['Designation'] + ' ' if exp['Designation'] is not None else ''
            else:
                cv_exp += exp['Designation/Organization'] + ' ' if exp['Designation/Organization'] is not None else ''
            cv_exp += exp['Description'] + ' ' if exp['Description'] is not None else ''

        cv_text += cv_exp
    else:
        if type(cv['Experience']) is list:
            if len(cv['Experience']) > 0:
                if type(cv['Experience'][0]) is str:
                    cv_exp = ' '.join(cv['Experience'])
    cv_text = cv_text.replace('\xa0', ' ')
    cv_words = preprocess_text(cv_text)
    sorted_cv_words = sorted(cv_words)
    cv_words = Counter(sorted_cv_words)
    return cv_words


def get_jd_cv_similarity(jd_words, cv_words):
    word_count = 0
    for cv_word in cv_words:
        if cv_word in jd_words:
            word_count += cv_words[cv_word]

    return word_count


def rank_cvs(jd, cvs_dict, cvs_df, check_location=False, check_job_title=False, check_required_experience_months=False):
    if type(jd) in [dict, pd.Series, pd.DataFrame]:
        jd_text = get_jd_text(jd)
        closest_cvs = cvs_df
        if check_location:
            closest_cvs = get_closest_location_cvs(cvs_df, jd['Location'])
        if check_job_title:
            closest_cvs = get_closest_job_title_cvs(closest_cvs, jd['Job Title'])
        if check_required_experience_months:
            required_experience_months = get_experience_months(jd['Required Experience'])
            closest_cvs = get_cvs_closest_experiences(closest_cvs, required_experience_months)
        sims = []
        for i in range(len(closest_cvs)):
            cv_id = closest_cvs.loc[i].ID
            cv_text = cvs_dict[cv_id]
            word_count = get_jd_cv_similarity(jd_text, cv_text)
            sims.append(word_count)

        closest_cvs['Similarity Score'] = sims
        closest_cvs = closest_cvs.sort_values('Similarity Score', ascending=False)
        closest_cvs.reset_index(drop=True, inplace=True)
        closest_cvs.pop('ID')
        return closest_cvs
    else:
        print('Unexpected type for Job Description (JD)')
        return