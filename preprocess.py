import csv
import pandas as pd

def checkABCD(str, char):
    if char == 'a':
        if 'a' in str and not 'b' in str and not 'c' in str and not 'd' in str:
            return True
    elif char == 'b':
        if not 'a' in str and 'b' in str and not 'c' in str and not 'd' in str:
            return True
    elif char == 'c':
        if not 'a' in str and not 'b' in str and 'c' in str and not 'd' in str:
            return True
    elif char == 'd':
        if not 'a' in str and not 'b' in str and not 'c' in str and 'd' in str:
            return True
    return False

def select_trian_data():
    data = []
    select = ['a','b','c','d']
    first = True
    # Select Train data 140 in total, 35 each for one label
    for char in select:
        with open('ChicagoFSWild.csv', newline='') as input_file:
            reader = csv.reader(input_file)
            column_name = next(reader)
            if first:
                column_name.append("Label")
                data.append(column_name)
                first = False
            count = 0
            for row in reader: 
                # in total 160 data, 40 each
                if count == 35:
                    break
                elif int(row[4]) <= 15 and row[10] == "train" and checkABCD(row[8], char):
                        count += 1
                        row.append(char)
                        data.append(row)

    with open('output_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

def select_dev_test_data():
    dev_test = ['dev','test']
    for name in dev_test:
        data = []
        select = ['a','b','c','d']
        first = True
        # Select Train data 140 in total, 35 each for one label
        for char in select:
            with open('ChicagoFSWild.csv', newline='') as input_file:
                reader = csv.reader(input_file)
                column_name = next(reader)
                if first:
                    column_name.append("Label")
                    data.append(column_name)
                    first = False
                count = 0
                for row in reader: 
                    # in total 30 data, 10 each
                    if count == 10:
                        break
                    elif int(row[4]) <= 15 and row[10] == name and checkABCD(row[8], char):
                            count += 1
                            row.append(char)
                            data.append(row)

        with open('output_{}.csv'.format(name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)
        
def combine_output():

    data = []
    name = ['train', 'dev', 'test']
    label_counts = {'a': [0, 0, 0], 'b': [0, 0, 0], 'c': [0, 0, 0], 'd': [0, 0, 0]}

    first = True
    for i, n in enumerate(name):
        with open(f'output_{n}.csv', newline='') as input_file:
            reader = csv.reader(input_file)
            column_name = next(reader)
            if first:
                data.append(column_name)
                first = False
            for row in reader:
                data.append(row)
                label = row[12]
                if label in label_counts:
                    label_counts[label][i] += 1

    sum_a = label_counts['a'][0] + label_counts['a'][1] + label_counts['a'][2]
    sum_b = label_counts['b'][0] + label_counts['b'][1] + label_counts['b'][2]
    sum_c = label_counts['c'][0] + label_counts['c'][1] + label_counts['c'][2]
    sum_d = label_counts['d'][0] + label_counts['d'][1] + label_counts['d'][2]

    with open('output_new.csv'.format(name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)


    print("Summary \n \
          Label A has total number: {}, with train data: {}, dev data: {}, and test data: {} \n \
          Label B has total number: {}, with train data: {}, dev data: {}, and test data: {} \n \
          Label C has total number: {}, with train data: {}, dev data: {}, and test data: {} \n \
          Label D has total number: {}, with train data: {}, dev data: {}, and test data: {} \
          ".format(sum_a,label_counts['a'][0],label_counts['a'][1],label_counts['a'][2],
                   sum_b,label_counts['b'][0],label_counts['b'][1],label_counts['b'][2],
                   sum_c,label_counts['c'][0],label_counts['c'][1],label_counts['c'][2],
                   sum_d,label_counts['d'][0],label_counts['d'][1],label_counts['d'][2]))


def main():
    # 1. Select subset data from current dataset
    # in total, there are 200 datas Train = 200 * 0.7 = 140
    # Dev = 200 * 0.15 = 30, Test = 200 * 0.15 = 30
    select_trian_data()
    select_dev_test_data()
    combine_output()


if __name__ == "__main__":
    main()