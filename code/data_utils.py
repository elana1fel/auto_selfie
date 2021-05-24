import os
import cv2
import selfie_utils
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def plot_res(smiling_data, not_smiling_data):
    '''
    This func plots the histogram of mar score for smiling and not smiling data 
   
    Input:
        smiling_data       list of mar scores for not smiling data
        not_smiling_data   list of mar scores for smiling data
    '''
    plt.hist(smiling_data, bins=20, alpha=0.2, label='smiling')
    plt.hist(not_smiling_data, bins=20, alpha=0.2, label='not smiling')

    plt.xlabel('mar')
    plt.title('histogram of smiling and not smiling mar score')
    plt.grid()
    plt.legend()
    plt.savefig('histogram_smiling_not_smiling_mar_score.png')



def data_loader(dataset_name, detector, predictor):
    '''
    This func loads the labels from the database and calculates mar result for each img.

   
    Input:
        dataset_name    str     database name
        detector        dlib obj
        predictor       dlib obj
    
    Output:
        smiling_data       list of mar scores for not smiling data
        not_smiling_data   list of mar scores for smiling data
    '''
    training_folder = 'databases/' + dataset_name + '/images'
    dataset = {}
    smiling_list = []
    not_smiling_list = []

    with open(os.path.join('databases', dataset_name, dataset_name + '_labels.json')) as labels_file:
        labels = json.load(labels_file)

    counter = 0
    for file in os.listdir(training_folder):
        counter += 1

        file_name = os.listdir(os.path.join(training_folder, file))
        for sub_file in file_name: #we may have more than one img in folder
            file_path = os.path.join(os.path.join(training_folder, file) ,sub_file)

            try:
                label = labels[sub_file]

            except:
                print(f"missing label for: {sub_file}")
                continue

            frame = cv2.imread(file_path)
            gray_img, _ = selfie_utils.edit_img(frame)
            face = selfie_utils.detect_face(gray_img, detector, predictor) 
            try:
                mar = face[0]['mar']

            except:
                mar = -1
                print(f"not detected face in: {sub_file}")
            
            if label:
                smiling_list.append(mar)

            else:
                not_smiling_list.append(mar)

    smiling_list.sort()
    not_smiling_list.sort()
    len(smiling_list), len(not_smiling_list)
    res = {'smiling' : smiling_list, 'not_smiling': not_smiling_list}
    jsonString = json.dumps(res)
    jsonFile = open("data.json", "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    #plot_res(smiling_list, not_smiling_list)
    return smiling_list, not_smiling_list


def train(dataset_name):
    '''
    This func finds the best bounds to mar score
    It uses percentile to determine mar upper and lower bounds

    Input:
        dataset_name    str     database name
    
    Output:
        mar_bounds      list of floats  the upper and lower bound of mar score
    '''
    detector, predictor = selfie_utils.init_model()
    smiling_list, not_smiling_list = data_loader(dataset_name, detector, predictor)

    #we checked that np.percentile(smiling_list, 31) = np.percentile(not_smiling_list, 69) + epsilon
    return np.percentile(smiling_list, 31)

def get_smile_status_mar_score(detector, predictor, gray_img, mar_score):
    '''
    Input:
    
    Output:
    '''
    face = selfie_utils.detect_face(gray_img, detector, predictor) 
    try:
        mar = face[0]['mar']

    except:
        return None

    if mar >= mar_score:
        return True
    return False


def validation(dataset_name, method='mar score', mar_score=0.31):
    '''
    This func tests the algorithms result vs the labels.

    Input:
        dataset_name    str     database name
        method          str     mar score or haar
    
    Output:
        precision       float   ratio of correctly predicted positive observations to the total predicted positive observations
        recall          float   ratio of correctly predicted positive observations to the all observations in actual class - yes
        f1_score        float   the weighted average of Precision and Recall. 
    '''
    print("running validation")
    
    if method == 'mar score':
        detector, predictor = selfie_utils.init_model()

    validation_folder = 'databases/' + dataset_name + '/images'

    smile_pos = 0
    not_smile_pos = 0
    smile_neg = 0
    not_smile_neg = 0
    smile = None

    with open(os.path.join('databases', dataset_name, dataset_name + '_labels.json')) as labels_file:
        labels = json.load(labels_file)

    counter = 0
    for file in os.listdir(validation_folder):
        counter += 1

        file_name = os.listdir(os.path.join(validation_folder, file))
        for sub_file in file_name: #we may have more than one img in folder
            file_path = os.path.join(os.path.join(validation_folder, file) ,sub_file)

            try:
                label = labels[sub_file]

            except:
                continue

            frame = cv2.imread(file_path)
            gray_img, _ = selfie_utils.edit_img(frame)

            if method == 'mar score':
                smile = get_smile_status_mar_score(detector, predictor, gray_img, mar_score)


            if smile is not None:
                if smile == label:
                    if smile:
                        smile_pos +=1

                    else:
                        not_smile_pos +=1
                
                else:
                    if smile:
                        smile_neg +=1

                    else:
                        not_smile_neg +=1

            elif smile is None:
                print(f"face not detected in {sub_file}")


    calculate_P_R_F([smile_pos, not_smile_pos, smile_neg, not_smile_neg])


def calculate_P_R_F(result_list):
    '''
    This func calculates precision, recall and f1 score.

    Input:
        result_list
    
    Output:
        precision       float   ratio of correctly predicted positive observations to the total predicted positive observations
        recall          float   ratio of correctly predicted positive observations to the all observations in actual class - yes
        f1_score        float   the weighted average of Precision and Recall. 
    '''
    # P : true_positives / (true_positives + false_positives)
    # R : true_positives / (true_positives + false_negative)

    print("calculating P R F1")
    smile_pos, not_smile_pos, smile_neg, not_smile_neg = result_list

    P = smile_pos/(smile_pos+smile_neg)
    R = smile_pos/(smile_pos+ not_smile_neg)
    
    if P+R:
        F= (2*P*R)/(P+R)
    else:
        F = 1

    print(f"P: {P}")
    print(f"R: {R}")
    print(f"F: {F}")
    return (P, R, F)

if __name__ == "__main__":
    dataset_name = 'lfw_dataset' #'lfw_dataset' or 'Selfie_dataset'
    mar_score = train(dataset_name)
    p, r, f1 = validation(dataset_name, mar_score=mar_score)