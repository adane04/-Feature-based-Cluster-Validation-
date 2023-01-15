#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
#warnings.simplefilter('ignore', ConvergenceWarning)

import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cluster import KMeans


# # Read   original data

# In[ ]:


data="Dataset_final_chronic.csv"
df=read_csv(data,index_col=0)
print(df.shape)
df.head(1)


# # Clustering  with Quality measure using  cross-Validation
# we can predict the probabilities of the 6 problems in patients in the test data based on their membership in the clusters. This can be done by using the crossvalidation procedure. In each fold of the crossvalidation:
# 
# a.  perform the clustering on the training part of the data set
# b.  assign patients from the test part of the data set obtained in step (a)
# c.  calculate a quality measure expressing the difference between the probabilities from the training data and the test data assigned to the same cluster in step (b)
# 
# The quality measures could be the Root Mean Squared Error (RMSE) or the Mean Average Percentage Error (MAPE), for example.

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import sqrt
import prince
from sklearn import preprocessing


# In[ ]:


cv = KFold(n_splits=10, random_state=42, shuffle=True)
for train_index, test_index in cv.split(df1):
    X_train, X_test = df1.iloc[train_index], df1.iloc[test_index]
print(len(X_train))
print(len(X_test))


# In[ ]:


RMSE_result=pd.DataFrame({})
MAPE_result=pd.DataFrame({})

km = KMeans(n_clusters=3,init='k-means++', max_iter=300, n_init=10) 
cv = KFold(n_splits=10, random_state=42, shuffle=True)
for train_index, test_index in cv.split(df1):
    X_train, X_test = df1.iloc[train_index], df1.iloc[test_index]
    #km.fit(X_train)
    #y_pred_train=km.predict(X_train)
    #y_pred_test=km.predict(X_test)
    X_train_df=pd.DataFrame(X_train)
    X_test_df=pd.DataFrame(X_test)
    print(X_train.shape)
    #print(X_train.tail())
    # Take only 58 features for both testing and training
    df_58_train= X_train_df[['sex','bmi','diastolic','sbp', 'hypertesion','diabetes','fatty ']]
    df_58_test= X_test_df[['sex','bmi','diastolic','sbp', 'hypertesion','diabetes','fatty ']]
    
    #  Apply MCA  for both trianing and testing 
    ca_train = prince.CA(n_components=2, n_iter=3,copy=True,check_input=True,engine='auto',random_state=3)
    ca_train= ca_train.fit(df_58_train)
    PC_train=ca_train.row_coordinates(df_58_train)
    PC_train.rename(columns={0:'Dim1',1:'Dim2',2:'Dim3'}, inplace=True)
    
    ca_test = prince.CA(n_components=2,n_iter=3,copy=True,check_input=True,engine='auto',random_state=42)
    ca_test = ca_test.fit(df_58_test)
    PC_test=ca_test.row_coordinates(df_58_test)
    PC_test.rename(columns={0:'Dim1',1:'Dim2',2:'Dim3'}, inplace=True)
    #print(PC_test.head())
    
    ## Normalize data  only training  before  model fitting  
    x = PC_train.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    PC_train_nor = pd.DataFrame(x_scaled)
    
    # model fitting  and predcitons 
    km.fit(PC_train_nor)
    y_pred_train=km.predict(PC_train)
    y_pred_test=km.predict(PC_test)
    #print (y_pred_train)
    #print (y_pred_test)
    
     # Calculate probabilities for  train
    X_train['cluster_train']=y_pred_train
    df_5_train=X_train[['sex','bmi','diastolic','sbp', 'hypertesion','diabetes','fatty ','cluster_train']]
    df_5_train.rename(columns={'sex','bmi','diastolic','sbp', 'hypertesion','diabetes','fatty ','cluster_train':'cl_train'}, inplace=True)
    df_sum_train=df_5_train.groupby(['cl_train']).sum()
    #col_sum_train=df_sum_train[df_sum_train.columns].sum(axis=0)
    #df_per_train=df_sum_train/986052 # calcualte the percentage of 1's in  each  cluster 
    cls_train=pd.DataFrame(y_pred_train)
    #cls.columns=['cluster_train']
    gp_train=cls_train.groupby(0).size()
    
    c0=df_sum_train.iloc[0]/gp_train.iloc[0]
    c1=df_sum_train.iloc[1]/gp_train.iloc[1]
    c2=df_sum_train.iloc[2]/gp_train.iloc[2]
    
    df_per_train=pd.concat([c0,c1,c2],axis=1,ignore_index=True)
    df_per_train=df_per_train.T
        
    # Calculate probabilities for  test
    X_test['cluster_test']=y_pred_test
    df_5_test=X_test[['sex','bmi','diastolic','sbp', 'hypertesion','diabetes','fatty ','cluster_test']]
    df_5_test.rename(columns={'sex','bmi','diastolic','sbp', 'hypertesion','diabetes','fatty ','cluster_test':'cl_test'}, inplace=True)
    df_sum_test=df_5_test.groupby(['cl_test']).sum()
    #col_sum_test=df_sum_test[df_sum_test.columns].sum(axis=0)
    #df_per_test=df_sum_test/109561 # calcualte the probabilities  of  outcomes  in  each  cluster 
    
    cls_test=pd.DataFrame(y_pred_test)
    #cls_test.columns=['cluster_test']
    gp_test=cls_test.groupby(0).size()
    
    c00=df_sum_test.iloc[0]/gp_test.iloc[0]
    c11=df_sum_test.iloc[1]/gp_test.iloc[1]
    c22=df_sum_test.iloc[2]/gp_test.iloc[2]
    
    df_per_test=pd.concat([c00,c11,c22],axis=1,ignore_index=True)
    df_per_test=df_per_test.T
    
    # Quality Measure  between training and testing
    for i in range (3):
        sum1=0
        #print('cluster:',i,'......................................')
        mse_sum=0
        mape_sum=0
        for k in range (6):
           
            #print(df_per_test)
            # calcualte  Mean squared error between training and testing 
            mse=(df_per_train.iloc[i,k] - df_per_test.iloc[i,k])**2
            mape=abs(df_per_train.iloc[i,k] - df_per_test.iloc[i,k])/abs(df_per_train.iloc[i,k])
            #print(df_per_test.columns[k])
            #print(df_per_train.columns[k],rms)
            #print(df_per_train.columns[k] ,':' ,mse.round(5))
            mse_sum=mse_sum+mse
        MSE=mse_sum/6  #6 is teh number of problems /columns 
        RMSE=sqrt(MSE)
                        
        mape_sum=mape_sum+mape
        MAPE=(mape_sum/6)*100
            
        RMSE_avg=pd.DataFrame([RMSE])
        RMSE_result=RMSE_result.append(RMSE_avg) 
        
        MAPE_avg=pd.DataFrame([MAPE])
        MAPE_result=MAPE_result.append(MAPE_avg)
        #print('mse_sum:',np.round(mse_sum,5))
        # print('mape_sum:',np.round(mape_sum,5))
        #print('RMSE:',np.round(RMSE,4))
        #print('MAPE:',np.round(MAPE,2),'%')
        
        # RMSE,  over all partitions 
        #RMSE_total=0
        #RMSE_total=RMSE_total+RMSE
        #RMSE_avg=RMSE_total/3   # 3 is the number of  clsuters
        
       
        #print('RMSE_avg:',RMSE_avg)
        
        #RMSE_avg=pd.DataFrame([RMSE_avg])
        #RMSE_result=RMSE_result.append(RMSE_avg)
        
      
      
        #MAPE_total=0
        #MAPE_total=MAPE_total+MAPE
        #MAPE_avg=MAPE_total/3 # # 3 is the number of  clusters
        
       
        #print('MAPE_avg:',MAPE_avg) 
        
        #MAPE_avg=pd.DataFrame([MAPE_avg])
        #MAPE_result=MAPE_result.append(MAPE_avg)
            


# ### Root Mean squared error

# In[ ]:


len(RMSE_result)


# In[ ]:


print('..........................................')
rmse1=RMSE_result.iloc[0:3] 
rmse1.columns = ["RMSE1"]

rmse2=RMSE_result.iloc[3:6]
rmse2.columns = ["RMSE2"]

rmse3=RMSE_result.iloc[6:9]
rmse3.columns = ["RMSE3"]

rmse4=RMSE_result.iloc[9:12] 
rmse4.columns = ["RMSE4"]

rmse5=RMSE_result.iloc[12:15]
rmse5.columns = ["RMSE5"]

rmse6=RMSE_result.iloc[15:18]
rmse6.columns = ["RMSE6"]

rmse7=RMSE_result.iloc[18:21] 
rmse7.columns = ["RMSE7"]

rmse8=RMSE_result.iloc[21:24]
rmse8.columns = ["RMSE8"]

rmse9=RMSE_result.iloc[24:27]
rmse9.columns = ["RMSE9"]

rmse10=RMSE_result.iloc[27:30]
rmse10.columns = ["RMSE10"]

#df_add = pd.concat([rmse1+rmse2+rmse3]+
rmse_add=pd.concat([rmse1,rmse2,rmse3,rmse4,rmse5,rmse6,rmse7,rmse8,rmse9,rmse10],axis=1, ignore_index=True)
rmse_add['RMSE_avg']=((rmse1['RMSE1']+rmse2['RMSE2']+rmse3['RMSE3']+rmse4['RMSE4']+
                      rmse5['RMSE5'] +rmse6['RMSE6']+rmse7['RMSE7']+rmse8['RMSE8']+
                      rmse9['RMSE9']+rmse10['RMSE10'])/10).round(5)
print(rmse_add.reset_index())


# ### Mean Absolute percent error

# In[ ]:


len(MAPE_result)


# In[ ]:


mape1=MAPE_result.iloc[0:3] 
mape1.columns = ["MAPE1"]

mape2=MAPE_result.iloc[3:6]
mape2.columns = ["MAPE2"]

mape3=MAPE_result.iloc[6:9]
mape3.columns = ["MAPE3"]

mape4=MAPE_result.iloc[9:12] 
mape4.columns = ["MAPE4"]

mape5=MAPE_result.iloc[12:15]
mape5.columns = ["MAPE5"]

mape6=MAPE_result.iloc[15:18]
mape6.columns = ["MAPE6"]

mape7=MAPE_result.iloc[18:21] 
mape7.columns = ["MAPE7"]

mape8=MAPE_result.iloc[21:24]
mape8.columns = ["MAPE8"]

mape9=MAPE_result.iloc[24:27]
mape9.columns = ["MAPE9"]
mape10=MAPE_result.iloc[27:30]
mape10.columns = ["MAPE10"]

#df_add = pd.concat([rmse1+rmse2+rmse3]+
mape_add=pd.concat([mape1,mape2,mape3,mape4,mape5,mape6,mape7,mape8,mape9,mape10],axis=1, ignore_index=True)
mape_add['MAPE_avg']=((mape1['MAPE1']+mape2['MAPE2']+mape3['MAPE3']+mape4['MAPE4']+mape5['MAPE5']+
                      mape6['MAPE6']+mape7['MAPE7']+mape8['MAPE8']+mape9['MAPE9']+mape10['MAPE10'])/10).round(2)
print(mape_add.reset_index())
print('..........................................')


# In[ ]:




