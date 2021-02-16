from sklearn.cluster import KMeans
#클러스터의 개수 지정(n개)
num_clusters = 3
#알맞은 매트릭스 Z 삽입
km = KMeans(n_clusters=num_clusters)
km.fit(Z)


import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(Z)
    kmeanModel.fit(Z)
    distortions.append(sum(np.min(cdist(Z, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / Z.shape[0])
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()