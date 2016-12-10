import csv
def read_data(fileName, target):
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            target.append([int(row[0]), int(row[1])])
