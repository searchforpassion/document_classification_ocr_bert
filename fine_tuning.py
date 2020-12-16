import cv2
from tqdm import tqdm
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
class_index_chosen = ['10','11']
rvl_dir = "F:/rvl-cdip"
test_file = rvl_dir + "/labels/test.txt"
save_directory = "F:/opt_zx/coproject_with_dy/save"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_name = "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)
model.train()
optim = AdamW(model.parameters(), lr=5e-5)

#decay options
# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
# optim = AdamW(optimizer_grouped_parameters, lr=5e-5)

# freezing the encoder
for param in model.base_model.parameters():
    param.requires_grad = False

def read_string_from_file(file_path, class_index_chosen):
    X_file_string = []
    y = []
    with open(file_path,'r') as f:
        for line in f:
            sample = line.split()
            if sample[1] in class_index_chosen:
                X_file_string.append(sample[0])
                y.append(sample[1])
    return X_file_string, y

test_X_string, test_y = read_string_from_file(test_file,class_index_chosen)
example_size = len(test_X_string)

def get_batch(X_file_string, y, batch_index, batch_size):
    raw_string_list = []
    y_label = []
    for i in range(batch_index, batch_index+batch_size):
        sample_file_path = rvl_dir + "/images/" + X_file_string[i]
        img_cv = cv2.imread(sample_file_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        document_string = pytesseract.image_to_string(img_rgb)
        raw_string_list.append(document_string)
        if y[i] == class_index_chosen[0]:
            y_label.append(0)
        else:
            y_label.append(1)
    batch_data = tokenizer(raw_string_list, truncation=True, padding=True, return_tensors="pt")
    return batch_data, torch.tensor(y_label)

batch_size = 8
loss_list = []
for epoch in range(1):
    print("=================EPOCH: " + str(epoch) + "===================")
    for batch_index in tqdm(range(int(example_size/batch_size))):
        optim.zero_grad()
        batch_data, y_label = get_batch(test_X_string, test_y, batch_index, batch_size)
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        labels = y_label.to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        if batch_index % 30 == 0:
            loss_list.append(loss.item())
print(loss_list)
print("Finished!")
# model.eval()

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)