import os
import json 

def parse(dataset_name):
    '''

    '''
    if dataset_name is 'lfw_dataset':
        parse_lfw_dataset(dataset_name)
    elif dataset_name is 'Selfie-dataset':
        parse_Selfie_dataset(dataset_name)

def parse_lfw_dataset(dataset_name):
    '''
    
    '''
    labels_dict = {}
    smile_file = os.path.join(dataset_name, 'SMILE_list.txt')
    non_smile = os.path.join(dataset_name, 'NON-SMILE_list.txt')
    for file_path in [smile_file, non_smile]:
        label_file = open(file_path, 'r')
        if file_path is smile_file:
            current_label = True
        elif file_path is non_smile:
            current_label = False

        Lines = label_file.readlines()
        for line in Lines:
            key = line.split('\n')[0]
            labels_dict[key] = current_label

    json_file = open(os.path.join(dataset_name, dataset_name + '_labels.json'), "w")
    json.dump(labels_dict, json_file, indent = 6)
    
    json_file.close()


def parse_Selfie_dataset(dataset_name):
    '''

    '''
    pass



if __name__ == "__main__":
    dataset_name = 'lfw_dataset' #'lfw_dataset' or 'Selfie-dataset'
    parse(dataset_name)
