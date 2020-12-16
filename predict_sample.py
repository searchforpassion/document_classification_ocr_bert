import cv2
example_class = '11'
example_index = 1

rvl_dir = "F:/rvl-cdip"
file_handle = rvl_dir + "/labels/test.txt"

cnt = 0
with open(file_handle,'r') as f:
    for line in f:
        sample = line.split()
        if sample[1] == example_class:
            if cnt == example_index:
                break
            cnt = cnt + 1

sample_file_path = rvl_dir + "/images/" + sample[0]
print(sample_file_path+'\n'+sample[1])

import pytesseract
from PIL import Image

img_cv = cv2.imread(sample_file_path)

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# we need to convert from BGR to RGB format/mode:
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
document = pytesseract.image_to_string(img_rgb)
print(document)
print("-----------------------------------------------")
print(len(document))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "bert-base-uncased"
# pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_sequence = tokenizer.tokenize(document)
# inputs = tokenizer(document)
print(tokenized_sequence)
print('tokenized sequence has '+str(len(tokenized_sequence))+' tokens.')
# print(inputs)
# print(len(inputs['input_ids']))

import matplotlib.pyplot as plt

# im = cv2.imread(sample_file_path)
img = plt.imshow(img_cv)
plt.show()