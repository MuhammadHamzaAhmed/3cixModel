# path
import os
#nltk
import re
#warning
import warnings
from os import listdir
from os.path import isfile, join
from shutil import rmtree
from datetime import date

# PDF and CSV
import PyPDF2
# import pikepdf
#spacy
import spacy
from fpdf import FPDF
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textract import process
warnings.filterwarnings('ignore')

class Converter:

    def __init__(self, read, save):
        self.read = read
        self.save = save
        self.pdf = [f for f in listdir(self.read) if isfile(join(self.read, f)) and f[-3:] == "pdf"]
        self.word = [f for f in listdir(self.read) if
                     isfile(join(self.read, f)) and f[-3:] == "doc" or f[-4:] == "docx"]
        self.processPdf()
        self.processWord()

    def processPdf(self):
        for pdf in self.pdf:
            try:
                pdfSaver = pikepdf.open(join(self.read, pdf))
                pdfSaver.save(join(self.save, pdf))
            except:
                pass

    def processWord(self):
        for word in self.word:
            try:
                texts = str(process(join(self.read, word))).split('\n')
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                for text in texts:
                    pdf.cell(200, 10, txt=text, align='L')
                pdf.output(join(self.save, word[:-4] + ".pdf"))
            except:
                pass


nlp = spacy.load("en_core_web_trf")
if "entity_ruler" in nlp.pipe_names:
    ruler = nlp.get_pipe("entity_ruler")
else:
    ruler = nlp.add_pipe("entity_ruler")
ruler.from_disk(r"jz_skill_patterns.jsonl")


class Parser:

    def __init__(self, pdfLink, required_skills, requiredLocation=None, requiredExperience=None):
        self.pdfLink = pdfLink
        self.cleanText = None
        self.cleanTextName = None
        self.cleanTextList = None
        self.cleanTextSkills = None
        self.skills = None
        self.requiredLocation = requiredLocation
        self.requiredExperience = requiredExperience
        self.name = ""
        self.email = ""
        self.country = ""
        self.experience = 0
        self.requiredSkills = required_skills
        self.ranked = 0

    def read(self):
        pdfFileObj = open(self.pdfLink, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict=False)
        numPage = pdfReader.numPages
        text = ""
        for i in range(numPage):
            text = text + pdfReader.getPage(i).extractText()
        pdfFileObj.close()
        text = text.split('\n')
        self.cleanTextList = ['']
        for data in text:
            if len(data) == 1 and data == "@":
                self.cleanTextList[-1] += data
            elif len(self.cleanTextList[-1]) > 1 and self.cleanTextList[-1][-1] == '@':
                self.cleanTextList[-1] += data
            elif len(data.strip()) >= 1:
                self.cleanTextList.append(data)
        self.cleanTextSkills = []
        for data in text:
            if len(data.strip()) >= 1:
                self.cleanTextSkills.append(data)
        self.cleanText = " ".join(self.cleanTextList)
        self.cleanTextName = " ".join(self.cleanTextSkills)

    def extractName(self):
        clean = self.cleanTextName
        clean = re.sub(r'[0-9]+', ' ', clean)
        clean = re.sub(r'[^\w]', ' ', clean)
        clean = re.sub(' +', ' ', clean)
        text_tokens = word_tokenize(clean)
        for i in range(len(text_tokens)):
            if len(text_tokens) > 1 and text_tokens[i][0] == 'n':
                text_tokens[i] = text_tokens[i][1:]
        text_tokens = [word.lower() for word in text_tokens if not word in stopwords.words() and len(word) > 2]
        text_tokens = [word.lower() for word in text_tokens if not word in stopwords.words() and len(word) > 2]
        file = open('../simpleWords.txt')
        removalList = file.read().split('\n')
        file.close()
        clean_new = []
        for word in text_tokens:
            if word not in removalList and word not in clean_new:
                clean_new.append(word)
        cleaned = " ".join(clean_new)
        english_nlp3 = spacy.load('en_core_web_lg')
        spacy_parser3 = english_nlp3(cleaned)
        for entity in spacy_parser3.ents:
            if entity.label_ == 'PERSON':
                self.name = entity.text.split(" ")
                if len(self.name) > 2:
                    self.name = self.name[:2]
                self.name = " ".join(self.name)
                break
        if len(self.name) == 0:
            cleaned = cleaned.split(' ')
            if len(cleaned) > 2:
                cleaned = cleaned[:2]
            self.name = " ".join(cleaned)
        if len(self.name) == 0:
            self.name = self.email

    def extractEmail(self):
        self.email = re.findall(r'[\w\.-]+@[\w\.-]+', self.cleanText)
        if self.email:
            self.email = self.email[0]
        else:
            self.email = ""
        if "com" in self.email:
            index = self.email.index('com')
            if index + 3 < len(self.email) and self.email[index + 3] != '.':
                self.email = self.email[:index + 3]
        elif "co" in self.email:
            index = self.email.index('co')
            if index + 2 < len(self.email) and self.email[index + 2] != '.':
                self.email = self.email[:index + 2]
        ind = re.search(r'[a-z]', self.email, re.IGNORECASE)
        if ind is not None:
            ind = ind.start()
            self.email = self.email[ind:]

    def extractLocation(self):
        file = open('../countries.txt')
        countries = file.read().split("\n")
        file.close()
        for country in countries:
            match = re.findall(country, self.cleanTextName)
            if match:
                self.country = match[0]
                break

    def extractSkills(self):
        # Getting Skills
        def get_skills(text):
            doc = nlp(text)
            myset = []
            subset = []
            for ent in doc.ents:
                if ent.label_ == "SKILL":
                    subset.append(ent.text)
            myset.append(subset)
            return subset

        clean = []
        for text in self.cleanTextSkills:
            review = re.sub(
                '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"',
                " ",
                text
            )
            review = review.lower()
            review = review.split()
            lm = WordNetLemmatizer()
            review = [
                lm.lemmatize(word)
                for word in review
                if not word in set(stopwords.words("english"))
            ]
            review = " ".join(review)
            clean.append(review)
        clean = " ".join(clean)
        self.skills = set(get_skills(clean))
        for skill in self.requiredSkills:
            if re.search(skill, self.cleanText, re.IGNORECASE):
                self.skills.add(skill)
        self.skills = list(self.skills)

    def extractExperience(self):
        exp = "".join(self.cleanText.split(" "))
        if re.search("experien", exp, re.IGNORECASE):
            index = re.search("experien", exp, re.IGNORECASE)
        elif re.search("employmen", exp, re.IGNORECASE):
            index = re.search("employmen", exp, re.IGNORECASE)
        else:
            return
        exp = exp[index.start():]
        exper = self.getExperience(exp)
        pattern = r'(((Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|June?|July?|Aug(ust)?|Sep(tember)?|Nov(ember)?|Dec(ember)?)|(\d{1,2}\/){0,2})[- ]?\d{4}?)'
        res = re.findall(pattern, exper)
        extractedDate = []
        if len(res) > 1:
            for tup in res:
                for ch in tup:
                    if len(ch) >= 4 and ch[-4:].isdecimal():
                        extractedDate.append(ch[-4:])
        if not extractedDate:
            extractedDate = re.findall("[0-9]{4}",data)
        if not extractedDate:
            return
        extractedDate = [int(i) for i in extractedDate]
        extractedDate.sort()
        today = date.today()
        cleanDates = []
        for dt in extractedDate:
            if dt < today.year:
                cleanDates.append(dt)
        if not cleanDates:
            return
        if re.search("-( +)?[a-z]+", exp, re.IGNORECASE) or re.search("till( +)?[a-z]+", exp, re.IGNORECASE):
            end = today.year
            start = cleanDates[0]
            self.experience = end- start
        else:
            end = cleanDates[-1]
            start = cleanDates[0]
            self.experience = end- start
        if len(str(self.experience)) > 2:
            self.experience = 0

    def getExperience(self, exp):
        tags = ['Education', "Role", "Responsib", "Award", 'Academic', 'Certifications', 'Trainings', 'Passport',
                'Personal']
        for tag in tags:
            if re.search(tag, exp, re.IGNORECASE):
                index = re.search(tag, exp, re.IGNORECASE)
                exp = exp[:index.start()]
        return exp

    def rankResume(self):
        line = " ".join(self.skills).lower()
        score = 0
        for skill in self.requiredSkills:
            if skill.lower() in line:
                score += 1
        total = len(self.requiredSkills)
        if self.requiredLocation is not None:
            if self.country == self.requiredLocation:
                score += 1
            total += 1
        if self.requiredExperience is not None:
            if self.requiredExperience <= self.extractExperience():
                score += 1
            total += 1
        self.ranked = round(score / total, 2)