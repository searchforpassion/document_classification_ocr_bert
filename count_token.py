import cv2

class_index_chosen = ['11']
rvl_dir = "F:/rvl-cdip"
test_file = rvl_dir + "/labels/test.txt"

def read_string_from_file(file_path, class_index_chosen):
    X_string = []
    y = []
    with open(file_path,'r') as f:
        for line in f:
            sample = line.split()
            if sample[1] in class_index_chosen:
                X_string.append(sample[0])
                y.append(sample[1])
    return X_string, y

test_X_string, test_y = read_string_from_file(test_file,class_index_chosen)
print('Files have been processed: '+str(len(test_X_string)))
print(test_X_string[0:5])

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

from transformers import AutoTokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

raw_string_len_list = []
raw_string_list = []
token_len_list = []

from tqdm import tqdm
for test_X_file in tqdm(test_X_string[95:100]):
    sample_file_path = rvl_dir + "/images/" + test_X_file
    # print(sample_file_path)
    img_cv = cv2.imread(sample_file_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    document_string = pytesseract.image_to_string(img_rgb)
    # print(document_string)
    raw_string_list.append(document_string)
    raw_string_len_list.append(len(document_string))

    # tokenized_sequence = tokenizer.tokenize(document_string)
    # print(tokenized_sequence)

tokenized_dict = tokenizer(raw_string_list, truncation=True, padding=True, return_tensors="pt")
print(raw_string_len_list)
print(tokenized_dict['input_ids'].shape)