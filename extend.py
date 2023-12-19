import os
import csv

folder_path = 'D:/HocMay/drive/test'

csv_file_path = 'test.csv'

with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(['file', 'result', 'predict', 'confidence'])
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            result = filename.split('_')[0]
            csv_writer.writerow([filename, result, '', ''])
            

print(f'Created CSV in: {csv_file_path}')
