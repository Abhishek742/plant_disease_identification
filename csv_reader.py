import csv
import json

def read_supplement_info():
    with open('resource_files/supplement_info.csv','r') as csv_file:
        supplements = {}
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            supplements[line[1]] = {
                "name" : line[2],
                "image_url" : line[3],
                "buy_link" : line[4]
            }
        print(json.dumps(supplements,indent = 4))

def read_disease_info():
    with open('resource_files/disease_info.csv','r') as csv_file:
        diseases = {}
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            # print(line[2])
            diseases[line[1]] =  str(line[3])
        print(json.dumps(diseases,indent = 4))

if __name__ == '__main__':
    read_disease_info()