
import json
import csv
 
 
# Opening JSON file and loading the data
# into the variable data
with open('game.json') as json_file:
    data = json.load(json_file)
 
game_data = data['phases']
 
# now we will open a file for writing
data_file = open('game.csv', 'w')
 
# create the csv writer object
csv_writer = csv.writer(data_file)
 
# Counter variable used for writing
# headers to the CSV file
count = 0
 
for phase in game_data:
    if count == 0:
 
        # Writing headers of CSV file
        header = phase.keys()
        csv_writer.writerow(header)
        count += 1
 
    # Writing data of CSV file
    csv_writer.writerow(phase.values())
 
data_file.close()
