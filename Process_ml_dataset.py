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
            #if(rowCount>100):break
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
    #df = pd.DataFrame({'Type': a,'Floor Size(SQFT)': f,'Number of Rooms': d,'Number of Bathroom': e,'Amount(BDT)': g})
    df.to_csv("processed_ml_dataset.csv", index=False)

    df = pd.read_csv('processed_ml_dataset.csv')
    print('Dimension of dataset= ', df.shape)
    df.head()
    from sklearn.cluster import KMeans
    df.dropna(axis=0, how='any', subset=['Lat', 'Lon'], inplace=True)
    # Variable with the Longitude and Latitude
    X = df.loc[:, ['Type', 'Lat', 'Lon']]
    #print(X.head(10))
    kmeans = KMeans(n_clusters=10, max_iter=1000, init='k-means++')
    lat_long = df.values[:, 1:3]
    #print(lat_long)
    # lot_size =  df.values[:, 2]
    # kmeans = KMeans(n_clusters = 3, init ='k-means++')
    kmeans.fit(lat_long)  # Compute k-means clustering.
    X['cluster_label'] = kmeans.fit_predict(lat_long)
    centers = kmeans.cluster_centers_  # Coordinates of cluster centers.
    labels = kmeans.predict(lat_long)  # Labels of each point
    #print(X.head(10))

    cluster = kmeans.fit_predict(lat_long)
    print(cluster)

    df = pd.DataFrame({'Type': a, 'Floor Size(SQFT)': f, 'Number of Rooms': d, 'Number of Bathroom': e, 'Cluster Label': cluster,'Amount(BDT)': g})
    df.to_csv("final_processed_ml_dataset.csv", index=False)
