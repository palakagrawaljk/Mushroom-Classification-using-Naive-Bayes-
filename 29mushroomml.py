
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


rdata=pd.read_csv("mushrooms.csv")


# In[ ]:


pdata=rdata[rdata['class']=='p']


# In[ ]:


pdata.head()#sparse data


# In[ ]:


edata=rdata[rdata['class']=='e']


# In[ ]:


edata.head()


# In[ ]:


pdata.shape


# In[ ]:


edata.shape


# In[ ]:


#we want 75% dat for training
traindata=int(.75*len(rdata))
traindata


# In[ ]:


etrain=edata.iloc[0:3046,1:]#we do not want 1st column
ptrain=pdata.iloc[0:3046,1:]
etest=edata.iloc[3046:,1:]
ptest=pdata.iloc[3046:,1:]
etest.shape


# In[ ]:


ptest.shape


# In[ ]:


etrain.shape


# In[ ]:


pcape=etrain.shape[0]/traindata


# In[ ]:


pcape


# In[ ]:


pcapp=ptrain.shape[0]/traindata


# In[ ]:


pcapp


# In[ ]:


traindata=pd.concat([etrain,ptrain],axis=0)


# In[ ]:


#uniquevalue=etrain['cap-shape'].unique()
#uniquevalue


# In[ ]:


#pcapex=ptrain[ptrain['cap-shape']=='x'].shape[0]/ptrain.shape[0]


# In[ ]:


#pcapex


# In[ ]:


seriesname=etrain.columns


# In[ ]:


seriesname


# In[ ]:


blanklist=list(map(lambda x: rdata[x].unique(),seriesname))
#for series in seriesname:
#    blanklist.append(etrain[series].unique())


# In[ ]:


blanklist


# In[ ]:


len(blanklist)


# In[ ]:


#epfeature


# In[ ]:



pdicf={}

for coln,array in zip(seriesname,blanklist):
    pbdic={}
    for values in array:
        pbdic[values]=ptrain[ptrain[coln]==values].shape[0]/ptrain.shape[0]
        if pbdic[values]==0:
            pbdic[values]=1/(ptrain.shape[0]+len(array))
    pdicf[coln]=pbdic
    
edicf={}

for coln,array in zip(seriesname,blanklist):
    ebdic={}
    for values in array:
        ebdic[values]=etrain[etrain[coln]==values].shape[0]/etrain.shape[0]
        if ebdic[values]==0:
            ebdic[values]=1/(etrain.shape[0]+len(array))
    edicf[coln]=ebdic
    


# In[ ]:


pdicf


# In[ ]:


edicf


# In[ ]:


def multivariatenaivebayesclassifier(oneexample):
    pfeatures_given_e=1
    pfeatures_given_p=1
    for i in seriesname:
        pfeatures_given_e=edicf[i][oneexample[i]]*pfeatures_given_e
        pfeatures_given_p=pdicf[i][oneexample[i]]*pfeatures_given_p
    num1=pfeatures_given_e*pcape
    num2=pfeatures_given_p*pcapp
    p_is_e_givenfeatures=num1/(num1+num2)
    return(p_is_e_givenfeatures)


# In[ ]:


truepositivecount=0
falsepositivecount=0
truenegativecount=0
falsenegativecount=0
#negative is e 
for i in range(0,len(etest)):
    pe_given_features=multivariatenaivebayesclassifier(etest.iloc[i,:])
    if pe_given_features >0.5:
        truenegativecount+=1 # or shi bataya, ki e h
    else:
        falsepositivecount+=1 #yeh galat bataya ,ki p h
print(truenegativecount)   
print(falsepositivecount)
        


# In[ ]:


for i in range(0,len(ptest)):
    pe_given_features=multivariatenaivebayesclassifier(ptest.iloc[i,:])
    if pe_given_features <0.5:
        truepositivecount+=1
    else:
        falsenegativecount+=1
print(truepositivecount)   #shi bataya,  ki p h
print(falsenegativecount)  #galat bataya , ki e h


# In[ ]:


precision=(truenegativecount+truepositivecount)/(etest.shape[0]+ptest.shape[0])*100
precision


# In[ ]:


recall=(truepositivecount/(ptest.shape[0]))
recall

