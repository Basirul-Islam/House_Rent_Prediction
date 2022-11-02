import numpy as np
import  pandas as pd
import  csv
import matplotlib.pyplot as plt

type = []
lat = []
lon = []
floorSize = []
floorSizeRefine = []
numberOfRooms = []
numberOfBathroom = []
amount = []

def read_csv():

    with open('ml_dataset.csv', mode = 'r') as file:
        csvFile = csv.reader(file)

        rowCount = 0
        for lines in csvFile:
            if(rowCount > 0):
                type.append(int(lines[0]))
                lat.append(float(lines[1]))
                lon.append(float(lines[2]))

                floorSizeNumber = ''

                for j in range (len(lines[3])):
                    if(lines[3][j] != ','):
                        floorSizeNumber += lines[3][j]

                #floorSize.append(int(floorSizeNumber))
                floorSizeRefine.append(int(floorSizeNumber))
                numberOfRooms.append(int(lines[4]))
                numberOfBathroom.append(int(lines[5]))
                amount.append(int(lines[6]))
            rowCount+=1
    # print(amount)
    # import matplotlib
    # matplotlib.use('TkAgg')
    # plt.title('floorSize vs Ammount')
    # plt.xlabel(r'Floor Size', fontsize=14)
    # plt.ylabel(r'Amount', fontsize=14)
    # plt.plot(numberOfRooms, amount)
    # plt.show()
    #
    # plt.title('Lat vs Ammount')
    # plt.xlabel(r'Lat', fontsize=14)
    # plt.ylabel(r'Amount', fontsize=14)
    # plt.scatter(lat, amount, color = 'red')
    # plt.show()
    #
    # plt.title('Lon vs Ammount')
    # plt.xlabel(r'Lon', fontsize=14)
    # plt.ylabel(r'Amount', fontsize=14)
    # plt.scatter(lon, amount, color='green')
    # plt.show()
    #
    # plt.title('numberOfRooms vs Ammount')
    # plt.xlabel(r'numberOfRooms', fontsize=14)
    # plt.ylabel(r'Amount', fontsize=14)
    # plt.scatter(numberOfRooms, amount, color='green')
    # plt.show()
    #
    # plt.title('numberOfBathroom vs Ammount')
    # plt.xlabel(r'numberOfBathroom', fontsize=14)
    # plt.ylabel(r'Amount', fontsize=14)
    # plt.scatter(numberOfBathroom, amount, color='blue')
    # plt.show()
    #
    # plt.title('Type vs Ammount')
    # plt.xlabel(r'Type', fontsize=14)
    # plt.ylabel(r'Amount', fontsize=14)
    # plt.scatter(type, amount, color='blue')
    # plt.show()

    # plt.title('Lat vs Ammount')
    # plt.xlabel(r'Lat', fontsize=14)
    # plt.ylabel(r'Amount', fontsize=14)
    #
    # fig = plt.figure(figsize= (10, 10))
    # ax = plt.axes(projection = '3d')
    #
    # ax.scatter3D(lat, lon, amount, color = 'green')
    # plt.show()

    a = np.array(type)
    b = np.array(lat)
    c = np.array(lon)
    d = np.array(numberOfRooms)
    e = np.array(numberOfBathroom)
    f = np.array(floorSizeRefine)
    g = np.array(amount)

    df = pd.DataFrame({'Type': a,'Lat': b,'Lon': c,'Floor Size(SQFT)': f,'Number of Rooms': d,'Number of Bathroom': e,'Amount(BDT)': g})
    df.to_csv("processed_ml_dataset.csv", index=False)