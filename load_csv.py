import csv

def readCSV(fp):#fp is filepath to csv (can be relative)
    with open(fp) as f:
        data = [{key: val for key, val in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]

    for item in data:
        item['X_Intercept_Left']    = int(item['X_Intercept_Left'])
        item['X_Intercept_Right']   = int(item['X_Intercept_Right'])
        item['Slope_Left']          = float(item['Slope_Left'])
        item['Slope_Right']         = float(item['Slope_Right'])
    return data