from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np

#dummy data
Z= pd.DataFrame(columns=['x','y'])
Z.loc[0] =[2,3]
Z.loc[1] =[2,11]
Z.loc[2] =[3,10]

#클러스터의 개수 지정(n개)
num_clusters = 4
#데이터셋 Z 삽입
km = KMeans(n_clusters=num_clusters)
km.fit(Z)



from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, init='k-means++').fit(Z)
    kmeanModel.fit(Z)
    distortions.append(sum(np.min(cdist(Z, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / Z.shape[0])
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
