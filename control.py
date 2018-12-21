import load_csv as lc
import train
import numpy as np

#get first command in list of commands, disregarding all the rest and returning only the first direction pressed (ie 'up' or 'left')
def getFirstCommand(command_list):
    parsed_commands = []
    for command in command_list:
        parsed_commands.extend(command.split(' '))
    if 'pressed' in parsed_commands: #check if anything was pressed (as opposed to just released)
        return parsed_commands[parsed_commands.index('pressed') - 1]
    else:
        return '' #no button pressed so take no action. TODO: consider ignoring these images or infering an input based on surrounding images/button released

def process_data_commands(data):
    for i,item in enumerate(data):
        firstCommand = getFirstCommand(item['Commands'].split(','))
        data[i]['Commands'] = firstCommand
    return data

if __name__ == '__main__':
    # data = lc.readCSV('PI_CAR_DATA/PI_Car_Runs.csv')
    data = lc.readCSV('../PI_CAR_DATA/PI_Car_Runs.csv')
    data = process_data_commands(data)

    # print data
    print '--------'
    print len(data)
    print '--------'
    print data[0]['Image File']
    print data[1]['Image File']
    print '--------'

    train.startTraining(data, data_in_memory=False, dataset='raw')
#raw images are 320x120 px
#processed images are 320x240 px
