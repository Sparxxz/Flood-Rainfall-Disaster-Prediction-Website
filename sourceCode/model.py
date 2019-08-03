import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer,MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
import plotly.plotly as py
import plotly
from sklearn.externals import joblib

import warnings
warnings.filterwarnings("ignore")
#get_ipython().run_line_magic('matplotlib', 'inline')
#fd-future data set
#validating-0 or 1 (0-tetsing ,1= future prediction)
def flood_classifier(filename,fd,validating=0):

	data1=pd.read_excel('data/'+filename+'.xlsx')

	# In[4]:
	data1.shape
	# In[5]:

	#Fillng null entries with mean of their respective columns
	for i in range(1,len(data1.columns)):
	    data1[data1.columns[i]] = data1[data1.columns[i]].fillna(data1[data1.columns[i]].mean())
	# In[6]:
	data1.describe()
	# In[7]:
	y=data1['Flood']
	# In[8]:
	for i in range(len(y)):
	    if(y[i] >= 0.1):
	        y[i]=1
	# In[9]:

	y=pd.DataFrame(y)

	data1.drop('Flood',axis=1,inplace=True)


	# In[10]:
	data1.head()
	# In[11]:
	data1.hist(figsize=(6,6));

	#Breaking Date column into timestamp

	d1=pd.DataFrame()
	d1["Day"]=data1['Date']
	d1['Months']=data1['Date']
	d1['Year']=data1['Date']
	data1['Date']=pd.to_datetime(data1['Date'])
	d1["Year"]=data1.Date.dt.year
	d1["Months"]=data1.Date.dt.month
	d1["Day"]=data1.Date.dt.day

	#----------------------Resampling
	#------------not working for piyush
	dx=pd.DataFrame()
	dx['Date']=data1['Date']
	dx['Discharge']=data1['Discharge']
	dx=dx.set_index(['Date'])
	yearly = dx.resample('Y').sum()

	plt.figure(figsize=(9,8))
	plt.xlabel('YEARS')
	plt.ylabel('Level')
	plt.title(filename+" : Year wise Trends")
	plt.plot(yearly,'--')

	#plt.plot(yearly,style=[':', '--', '-'],title='Year wise Trends')
	plt.savefig('static/img/flood.png')
	#--------------------------------


	# In[18]:
	data1.drop('Date',inplace=True,axis=1)
	# In[19]:


	#Scaling the data in range of 0 to 1

	# Scaler=MinMaxScaler(feature_range=(0, 1))
	# Transform=Scaler.fit_transform(data1)
	# # In[20]
	# #Transform
	# # In[21]:
	# Transform=pd.DataFrame(Transform,columns=['Discharge','flood runoff','daily runoff','weekly runoff'])

	# # In[22]:
	# data1=Transform
	# In[23]:
	data1=pd.concat([d1,data1],axis=1)
	data1.head()

	#-----------------------for taking data upto 2015 as training and rest for testing------------------------------------------------
	locate=0;
	for i in range(len(data1["Day"])):
	    if(data1["Day"][i]==31 and data1["Months"][i]==12 and data1["Year"][i]==2015):
	        locate=i;
	        break;
	        
	i=locate+1
	print(i)

	x_train=data1.iloc[0:i,:]
	y_train=y.iloc[0:i]
	x_test=data1.iloc[i:,:]
	y_test=y.iloc[i:]


	# In[25]:


	x_train.drop(labels=['Day','Months','Year'],inplace=True,axis=1)
	x_test.drop(labels=['Day','Months','Year'],inplace=True,axis=1)


	# In[26]:


	# nl=Normalizer()
	# x_train=nl.fit_transform(x_train)
	# x_test=nl.transform(x_test)


	# In[27]:


	# y_train=nl.transform(y_train)
	# y_test=nl.transform(y_test)


	# In[28]:


	#-----------------Upsampling the data (as very less entries of flood =1 is present)-----------------
	sm = SMOTE(random_state=2)
	X_train_res, Y_train_res = sm.fit_sample(x_train, y_train)
	# In[29]:
	X_train_res.shape
	# In[30]:
	x_train, y_train = shuffle( X_train_res, Y_train_res, random_state=0)

	x_train.shape,x_test.shape,y_train.shape,y_test.shape


	# In[32]:


	# #---------------Logistic Regression--------------------------
	# from sklearn.linear_model import LogisticRegression
	# reg=LogisticRegression()
	# reg.fit(x_train,y_train)
	# y_predict1=reg.predict(x_test)
	# print(set(y_predict1))
	# print(reg.score(x_train,y_train))
	# print(reg.score(x_test,y_test))
	# print(classification_report(y_test, y_predict1))
	# print("mean_absolute_error=",mean_absolute_error(y_test, y_predict1))


	# # In[34]:

	#-----------------------LinearDiscriminantAnalysis---------------------------------
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

	# clf1=LinearDiscriminantAnalysis()
	# clf1.fit(x_train,y_train)

	#-----------------------saving & Loading the model-------------------------------------------

	path='trained/'+filename+'_LDA'
	#joblib.dump(clf1, path+'.pkl')
	clf1= joblib.load(path+'.pkl')

	#---------------------------------------------------------------------------------

	y_predict3=clf1.predict(x_test)
	print(set(y_predict3))
	print(clf1.score(x_train,y_train))
	print(clf1.score(x_test,y_test))
	print(classification_report(y_test, y_predict3))
	mae=mean_absolute_error(y_test, y_predict3)
	print("mean_absolute_error=",mae)

	# In[35]:
	#---------------------------KNeighborsClassifier------------------------------------------

	# from sklearn.neighbors import KNeighborsClassifier

	# clf2=KNeighborsClassifier()
	# clf2.fit(x_train,y_train)
	# y_predict4=clf2.predict(x_test)
	# print(set(y_predict4))
	# print(clf2.score(x_train,y_train))
	# print(clf2.score(x_test,y_test))
	# print(classification_report(y_test, y_predict4))
	# print("mean_absolute_error=",mean_absolute_error(y_test, y_predict4))

	# # In[36]:

	#-------------------------------Testing-----------------------------------------------
	# In[38]:
	data1.head()
	# In[39]:
	def predicting(future_data):
		# xx=[13214.0,0.0,0.36,2.08]
		#xx=[4990.0,0.0,1.40,15.38]
		xx=future_data
		xx=np.array(xx)
		xx=xx.reshape((-1, 4))
		xx=clf1.predict(xx)
		# xx=reg.predict(xx)
		return xx
	xx=predicting(fd)
	return xx,mae
#xx=predicted value of flood 0 or 1


