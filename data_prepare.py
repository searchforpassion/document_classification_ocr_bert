

rvl_dir = "F:/rvl-cdip"
train_file = rvl_dir + "/labels/train.txt"
val_file = rvl_dir + "/labels/val.txt"
test_file = rvl_dir + "/labels/test.txt"

class_index_dic = {"0":"letter","1":"form","2":"email","3":"handwritten","4":"advertisement","5":"scientific report","6":"scienttific publication","7":"specification","8":"file folder","9":"news article","10":"budget","11":"invoice","12":"presentation","13":"questionnaire","14":"resume","15":"memo"}
class_index_chosen = ['1','10','11']

def print_chosen_class(chosen_list):
    for key in chosen_list:
        print("{} {}".format(str(key), class_index_dic[str(key)]))

print_chosen_class(class_index_chosen)

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


sample_file = sample[0]
sample_file_path = rvl_dir + "/images/" + sample_file
print(sample_file_path)
print(sample[1])
    

