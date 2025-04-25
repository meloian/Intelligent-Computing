import csv

def _yn_to_bool(value):
    # convert y/n and similar to bool
    v = value.strip().lower()
    if v == 'y' or v == 'yes' or v == 'true' or v == '1':
        return True
    return False

def input_params():
    # prompt for parameters
    lux = float(input("Зовнішня освітленість, lx: "))
    hour = float(input("Час доби [0-23.99], год: "))
    pres = input("Присутність у кімнаті (y/n): ")
    presence = _yn_to_bool(pres)
    eco = input("Еко-режим (y/n): ")
    eco_mode = _yn_to_bool(eco)
    params = {}
    params['lux'] = lux
    params['hour'] = hour
    params['presence'] = presence
    params['eco'] = eco_mode
    return params

def load_csv(path):
    # read csv into list of dicts
    rows = []
    f = open(path, 'r', encoding='utf-8', newline='')
    reader = csv.DictReader(f)
    for row in reader:
        rec = {}
        rec['lux'] = float(row['lux'])
        rec['hour'] = float(row['hour'])
        rec['presence'] = _yn_to_bool(row['presence'])
        rec['eco'] = _yn_to_bool(row['eco'])
        rows.append(rec)
    f.close()
    return rows

def save_results_csv(path, records):
    # write results to csv
    if len(records) == 0:
        print('no records to save')
        return
    header = list(records[0].keys())
    f = open(path, 'w', encoding='utf-8', newline='')
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)
    f.close()
    print('[ok] results stored -> ' + str(path))