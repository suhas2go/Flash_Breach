
# coding: utf-8

# # Vehicle Routing Problem using Machine Learning

# # Solution Approach

# ### Implemented successfully: Clustering with constraint on cluster weight - K means with suitable k and then visited cluster and the points within the cluster in a particular order
# ### We observed that the given are fairely uniform distributed both in terms of spatial and fuel terms. Therefore, we built clusters, a little more than the minimum of 136516/975 = 3272. We generated 4000 clusters with constraint on the cluster fuel weight to be not more than 975. 
# ### We chose to visit the clusters in the order of the distance between their centroids and the origin. Within each cluster we visited first the point which was closest to the origin and then we visited the points closest to the current point and so on. 

# ## Import dataset

# In[1]:

import pandas as pd
df=pd.read_csv("../data/Breach.csv")
df.head()


# In[3]:

list(df.columns.values)


# In[4]:

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
get_ipython().magic(u'matplotlib inline')
import numpy as np

X=np.array(df.drop(['Breach No.','Fuel weight'],1))


# In[5]:

df.shape


# # Custom K Means Algorithm implemented with constraint on the total fuel weight a cluster can hold

# In[6]:

import sys
class CustomKMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.cluster_weights=[0]*k;
    def fit(self,data,fuel_weight):
        self.labels=[0]*len(data)
        
        
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            z=0;
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                t=1;
                limit=1
                while(t):
                    if limit>self.k:
                        print "indicates increase of clusters"
                        sys.exit()

                    classification = distances.index(min(distances))
                    if self.cluster_weights[classification]+fuel_weight[z]<=975:
                        self.cluster_weights[classification]+=fuel_weight[z]
                        self.classifications[classification].append(featureset)
                        self.labels[z]=classification;
                        t=0
                    else:
                        distances[classification]=float("inf")
                        limit=limit+1
                z=z+1;

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break
    


# ## We observed that the distribution of points is uniform, therefore to save time we will not update the centroids max_iter=1

# In[7]:

nclf=CustomKMeans(k=4000,max_iter=1)


# In[8]:

nclf.fit(X,df['Fuel weight'].tolist())


# In[29]:

nclf.classifications[0]


# In[11]:

nnnlabels=nclf.labels


# In[18]:

se=pd.Series(nnnlabels)
ndf=df
ndf['cluster']=se.values


# In[19]:

ndf


# In[21]:

ndf.to_csv('..data/output/output1.csv') #taking backup


# In[22]:

temp=ndf


# In[137]:

temp.sort_values("cluster")


# ##Check if all clusters satisfy constraint

# In[27]:

ctr=0
for i in range(4000):
    if ndf.loc[ndf['cluster']==i,'Fuel weight'].sum()<975:
        ctr+=1
print ctr


# ##Generate Columns necessary for traversal

# In[34]:

from math import *
def dist_org(row):
    ans=pow(row['X coordinate'],2)+pow(row['Y coordinate'],2)
    return sqrt(ans)
temp['orgdist']=temp.apply(lambda row: dist_org(row),axis=1)


# In[35]:

temp.head()


# In[36]:

centroids=nclf.centroids


# In[51]:

def is_centroid(row):
    return row['Breach No.']<=4000   
temp['centroid']=temp.apply(lambda row:is_centroid(row),axis=1)


# In[54]:

temp


# In[55]:

temp.to_csv('..data/output/output2.csv')


# In[63]:

c=temp.sort_values(by=["centroid","orgdist"],ascending=[False,True])


# In[65]:

c.index=range(len(X))


# ## After all preprocessing done above, we are have finally obtained the data frame required to make Flash traverse and close the breaches!

# In[67]:

c.head()


# ## Traversing the points
# ### We will visit the cluster whose centroid is closest to the origin and then we will visit the points inside it in
# ### the following order: 1st visiting the closest point to the origin and then visiting the closest point to our 
# ### current point and so on

# In[71]:

cluster_order=c.loc[c['centroid']==True,'cluster'].tolist()


# In[115]:


def dist_from_curr(row,curr_point):
    ans=pow(row['X coordinate']-curr_point['X coordinate'],2)+pow(row['Y coordinate']-curr_point['Y coordinate'],2)
    return sqrt(ans)
    
final_df=pd.DataFrame(index=np.arange(0,len(X)),columns=('Breach No.','Trip No.'))
k=0
for i in range(4000):
    curr_cluster=c.loc[c['cluster']==cluster_order[i]]
    c2=curr_cluster.sort_values('orgdist')
    c2.index=range(len(c2))
    
    first_visit=c2.ix[0]
    final_df.loc[k]=[first_visit['Breach No.'],i+1]
    k=k+1
    curr_point=first_visit
    c2=c2[c2['Breach No.']!=curr_point['Breach No.']]
    c2.index=range(len(c2))
    x=c2.apply(lambda row:dist_from_curr(row,curr_point),axis=1)
    while not(c2.empty):
        #generate next point to visit
        c2.index=range(len(c2))
        x=c2.apply(lambda row:dist_from_curr(row,curr_point),axis=1)
        y=x.tolist()
        next_index=y.index(min(y))
        next_point=c2.ix[next_index]
        final_df.loc[k]=[next_point['Breach No.'],i+1]
        k=k+1
        curr_point=next_point
        c2=c2[c2['Breach No.']!=curr_point['Breach No.']]
final_df     


# In[86]:

cd=curr_cluster.sort_values('orgdist')
cd.index=range(len(cd))
cd


# In[89]:

asfd=cd.ix[0]
asfd['cluster']


# ## One breach to be left open
# ### Since the cost function is solely dependent on distance, we will choose to leave out the furthest cluster's last point

# In[131]:

ffinal_df=final_df[final_df['Breach No.']!=21242]
ffinal_df


# In[132]:

ffinal_df.to_csv('../data/output/final_submission.csv',index=False)


# # Evaluation of the algorithm

# In[120]:

def dist_from_curr(row,curr_point):
    ans=pow(row['X coordinate']-curr_point['X coordinate'],2)+pow(row['Y coordinate']-curr_point['Y coordinate'],2)
    return sqrt(ans)
    
eval_df=pd.DataFrame(index=np.arange(0,len(X)),columns=('Breach No.','Trip No.','X coordinate','Y coordinate'))
k=0
for i in range(4000):
    curr_cluster=c.loc[c['cluster']==cluster_order[i]]
    c2=curr_cluster.sort_values('orgdist')
    c2.index=range(len(c2))
    
    first_visit=c2.ix[0]
    eval_df.loc[k]=[first_visit['Breach No.'],i+1,first_visit['X coordinate'],first_visit['Y coordinate']]
    k=k+1
    curr_point=first_visit
    c2=c2[c2['Breach No.']!=curr_point['Breach No.']]
    c2.index=range(len(c2))
    x=c2.apply(lambda row:dist_from_curr(row,curr_point),axis=1)
    while not(c2.empty):
        #generate next point to visit
        c2.index=range(len(c2))
        x=c2.apply(lambda row:dist_from_curr(row,curr_point),axis=1)
        y=x.tolist()
        next_index=y.index(min(y))
        next_point=c2.ix[next_index]
        eval_df.loc[k]=[next_point['Breach No.'],i+1,next_point['X coordinate'],next_point['Y coordinate']]
        k=k+1
        curr_point=next_point
        c2=c2[c2['Breach No.']!=curr_point['Breach No.']]
eval_df  


# ## Calculating the cost function

# In[128]:

def dist_points(x,y):
    ans=pow(x[0]-y[0],2)+pow(x[1]-y[1],2)
    return sqrt(ans)
k=1
i=0
cost=0
while (i<len(X)-1):
    prev=[0,0]
    while (i<len(X)-1) and (eval_df.ix[i]['Trip No.']==k):           
            curr=[eval_df.ix[i]['X coordinate'],eval_df.ix[i]['Y coordinate']]
            cost+=dist_points(prev,curr)
            prev=curr
            i=i+1
    k=k+1
print cost
    

