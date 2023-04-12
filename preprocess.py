import csv

def checkABCD(str):
    if 'a' in str and not 'b' in str and not 'c' in str and not 'd' in str:
        return True
    if not 'a' in str and 'b' in str and not 'c' in str and not 'd' in str:
        return True
    if not 'a' in str and not 'b' in str and 'c' in str and not 'd' in str:
        return True
    if not 'a' in str and not 'b' in str and not 'c' in str and 'd' in str:
        return True
    return False

def process_select_data():
    with open('ChicagoFSWild.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # read first row only
        column_name = next(reader)
        data = []
        data.append(column_name)
        for row in reader:
            if int(row[4]) <= 8 and checkABCD(row[8]):
                data.append(row)
    print(len(data))

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

def process_separate_type():
    a = 0
    b = 0
    c = 0
    d = 0
    data_train = []
    data_dev = []
    data_test = []
    with open('output.csv', newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=';')
        # read first row only
        column_name = next(reader)
        data_train.append(column_name)
        data_dev.append(column_name)
        data_test.append(column_name)
        for row in reader:
            if row[12] == 'a':
                a += 1
            elif row[12] == 'b':
                b += 1
            elif row[12] == 'c':
                c += 1
            elif row[12] == 'd':
                d += 1
            if row[10] == "train":
                data_train.append(row)
            if row[10] == "dev":
                data_dev.append(row)
            if row[10] == "test":
                data_test.append(row)
    print("a: ",a," b: ",b," c: ",c," d: ",d)


    with open('output_train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data_train:
            writer.writerow(row)
    print("numbe lines train:",len(data_train))
    
    with open('output_dev.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data_dev:
            writer.writerow(row)
    print("numbe lines dev:",len(data_dev))

    with open('output_test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data_test:
            writer.writerow(row)
    print("numbe lines test:",len(data_test))



def main():
    # 1. Select subset data from current dataset
    # process_select_data()
    # 2. Seperate data to train, dev, and test
    process_separate_type()

if __name__ == "__main__":
    main()