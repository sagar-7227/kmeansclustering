import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset =pd.read_csv('machinelearning.csv')
# print(len(dataset))

x=dataset.iloc[:,3:5].values
# print(x)
wcs =[]
for i in range(1,11):
    kmeans =KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(x)
    wcs.append(kmeans.inertia_)

# print(wcs)
# print(plt.plot(range(1,11),wcs))
# plt.show()


kmeans=KMeans(n_clusters=5,init='k-means++')
y_kmeans = kmeans.fit_predict(x)
# print(y_kmeans)

# print(pd.concat([dataset,pd.DataFrame(y_kmeans)],axis=1))
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='blue')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='red')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='pink')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='yellow')
plt.show()



