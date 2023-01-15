#!/usr/bin/env python
# coding: utf-8

# In[1]:

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


# # Cluster 1

# read all the data  for clluster 1
data_zero="Cluster_1.csv"
df0=read_csv(data_zero,index_col=0)
print(df0.shape)
df0.head(3)
df00 = df0[['hyper','diabet','fatty']]
print(df00.shape)
df00.head(1)


# # Cluster 2
# read all the data  from a cluster
data_one="Cluster_2.csv"
df1=read_csv(data_one,index_col=0)
print(df1.shape)
df1.head(1)
df11 = df1[['hyper','diabet','fatty']]
print(df11.shape)
df11.head(2)


# # Cluster 3
# read all the data  for morto
data_two="Cluster_3.csv"
df2=read_csv(data_two,index_col=0)
print(df2.shape)
df2.head(1)
df22 = df2[['hyper','diabet','fatty']]
print(df22.shape)
df22.head(2)


# # Chi-Sqaure Test between clusters
# #### testing Observed values for  three clusters with variable hypertension
df0_table = pd.crosstab(index=df00['hyper'], columns="count")
df0_table.T # tain table
df1_table = pd.crosstab(index=df11['hyper'], columns="count")
df1_table.T #Test table
df2_table = pd.crosstab(index=df22['hyper'], columns="count")
df2_table.T #Test table


# Join tables
tb= pd.concat([df0_table,df1_table,df2_table],axis=1,ignore_index=True) #Join the two tables
tb.columns = ["c0","c1","c2"]
#observed = tb.ix[0:len(df0_table),0:3]
observed = tb.iloc[0:len(df0_table),0:3]
observed

# Calculate row total and column total
col_total_obs=observed.pivot_table(index=observed.index, margins=True, margins_name='col_totals', aggfunc=sum)
col_total_obs['row_totals'] = col_total_obs[observed.columns].sum(axis=1)
col_total_obs


# # Expected Values for the three clusters
expected =  np.outer(col_total_obs["row_totals"][0:len(df0_table)],col_total_obs.loc["col_totals"][0:3]) / 19888
expected = pd.DataFrame(expected)

expected.columns = ["c0","c1","c2"]
expected.index = observed.index
expected
row=len(expected)-1 ;row
col=expected.shape[1]-1 ;col
tot=row*col
tot


# Chi-Square statstic
chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()
print(chi_squared_stat.round(4))
crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df =tot)   # *

print("Critical value")
print(crit.round(4))
p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                            df=tot)
print("P value")
print(p_value.round(4))


# ## the Complete code goes here
df_report2=pd.DataFrame({})
#df_report = pd.DataFrame(df_report, columns = ['in','var',"samp1_n", "samp1_per", "samp2_n", "samp2_tot","Chi-sq","DF","CV","PV"])
for i in range(0,3):
    df0_table = pd.crosstab(index=df00.iloc[:,i], columns="")
    df1_table = pd.crosstab(index=df11.iloc[:,i], columns="") 
    df2_table = pd.crosstab(index=df22.iloc[:,i], columns="") 
    # join tables
    tb = pd.concat([df0_table,df1_table,df2_table],axis=1, ignore_index=True)
    tb.columns = ["c0","c1","c2"]
    # observed
    #tb.columns = ["s","p"]
    observed = tb.iloc[0:len(df0_table),0:3]
    #row total and column total
    col_total_obs=observed.pivot_table(index=observed.index, margins=True, margins_name='col_totals', aggfunc=sum)
    col_total_obs['row_totals'] = col_total_obs[observed.columns].sum(axis=1)
    col_total_obs
    #calculate expected
    expected =  np.outer(col_total_obs["row_totals"][0:len(df0_table)],col_total_obs.loc["col_totals"][0:3])/19888
    expected = pd.DataFrame(expected)
    expected.columns = ["c0","c1","c2"]
    #expected.index = observed.index
    

    # Chi-Square Test
    chi_x = (((observed-expected)**2)/expected).sum().sum()
    chi_x=pd.DataFrame([chi_x]).round(4)
    
    #Critical Value 
    row=len(expected)-1 ;
    col=expected.shape[1]-1 ;
    tot=row*col
    crit = stats.chi2.ppf(q = 0.95,df = tot)
    crit=pd.DataFrame([crit]).round(4)
    
    # P-value 
    p_value = 1 - stats.chi2.cdf(x=chi_x, df=tot)
    p_value=pd.DataFrame(p_value).round(4)
    
    #Degrees of fredom
    dfree=tot
    dfree=pd.DataFrame([dfree]).round(4)
    
    # cluster 0 (count and percentage)
    val=df00.iloc[:,i].value_counts()
    per=(100. * (df00.iloc[:,i]).value_counts() / len(df00.iloc[:,i])).round(4)
    dfn = pd.concat([val, per], keys=['', ''],axis=1, sort=True) 
    
    # cluster 1 (count and percentage)
    val=df11.iloc[:,i].value_counts()
    per=(100. * (df11.iloc[:,i]).value_counts() / len(df11.iloc[:,i])).round(4)
    dfm = pd.concat([val, per], keys=['', ''],axis=1, sort=True)
    
    # cluster 2 (count and percentage)
    val=df22.iloc[:,i].value_counts()
    per=(100. * (df22.iloc[:,i]).value_counts() / len(df22.iloc[:,i])).round(4)
    dfc = pd.concat([val, per], keys=['', ''],axis=1, sort=True)
    
    #get variable names (column names)
    col=list(df00.columns.values)
    var_name=col[i]
    var_name=pd.DataFrame([var_name])
    #
    # Total row /data size for the sample and total sample
    samp0 = "%.0f (%.2f)" % ((len(df00)),len(df00)/(len(df00)+len(df11)+len(df22)))
    samp1 = "%.0f (%.2f)" % ((len(df11)),len(df11)/(len(df00)+len(df11)+len(df22)))
    samp2 = "%.0f (%.2f)" % ((len(df22)),len(df22)/(len(df00)+len(df11)+len(df22)))
    samp0=pd.DataFrame([samp0])
    samp1=pd.DataFrame([samp1])
    samp2=pd.DataFrame([samp2])
    # Join Results
    dfk = pd.concat([var_name,dfn,samp0,dfm,samp1,dfc,samp2,chi_x,dfree,crit,p_value],axis=1, sort=True,ignore_index=True).T.drop_duplicates().T
    #dfk.reset_index()
    #dfk[1] = dfk.index #make the variables as index
    #dfk.index.rename('foo', inplace=True)
    #dfk.reset_index(level=['foo'])
    
    
    #dfk.rename(columns={0:'Num_samp',1:'Num_samp',2:'Num_tot',3:'Per_tot',4:'X^2',5:'DF',6:'CV',7:'PV'}, inplace=True)
    dfk.rename(columns={0:'',1:'',2:'',3:'',4:'',5:'',6:'',7:'',8:'',9:'',10:''}, inplace=True)
    #df_final=dfk.columns=["Num_samp", "Num_samp", "Num_tot", "Per_tot","Chi-sq","DF","CV","PV"]
    #result.append(df_final)
    df_final=dfk.fillna('')
    df_report2=df_report2.append(df_final)
    #print(df_final)
print(df_report2)
#df_report2.to_csv("C:/Progetti/selected_negative_and_positive_samples/report_detail.csv")
#df3=pd.DataFrame(result)
#print(result)


writer = pd.ExcelWriter('Pvalue_detail_Km_chronic_dataset.xlsx')
df_report2.to_excel(writer,'Sheet1')
#df2.to_excel(writer,'Sheet2')
writer.save()


# In[ ]:




