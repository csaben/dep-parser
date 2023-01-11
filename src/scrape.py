import requests
import PyPDF2
url = "https://media.defense.gov/2020/Sep/01/2002488689/-1/-1/1/2020-DOD-CHINA-MILITARY-POWER-REPORT-FINAL.PDF"
response = requests.get(url)

# open the pdf file
with open('../input/pdf_txt/china_military_report.pdf', 'wb') as f:
    f.write(response.content)

# open the pdf using PyPDF2
with open('../input/pdf_txt/china_military_report.pdf', 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # extract the text from all the pages
    text = ""
    for i in range(len(pdf_reader.pages)):
        # text += pdf_reader.getPage(i).extractText()
        text += pdf_reader.pages[i].extract_text()
        
# write the text to a file
with open("../input/pdf_txt/china_military_report.txt", "w") as file:
    file.write(text)
