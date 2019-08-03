import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
#import mpld3

# ## Types of graphs
# - Bar graphs showing distribution of amount of rainfall.
# - Distribution of amount of rainfall yearly, monthly, groups of months.
# - Distribution of rainfall in subdivisions, districts form each month, groups of months.
# - Heat maps showing correlation between amount of rainfall between months.
# 

# In[2]:
def rainfall(year,region):
    data = pd.read_csv('data/Sub_Division_IMD_2017.csv')
    data = data.fillna(data.mean())
    data.info()


    # In[3]:

    data.head()


    # In[4]:

    data.describe()


    # In[5]:

    # data.hist(figsize=(12,12));


    # ## Observations
    # - Above histograms show the distribution of rainfall over months.
    # - Observed increase in amount of rainfall over months July, August, September.

    # In[6]:

    # data[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
    #        'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("YEAR").sum().plot(figsize=(13,8));


    # In[7]:

    # data[['YEAR','JF', 'MAM',
    #        'JJAS', 'OND']].groupby("YEAR").sum().plot(figsize=(13,8));


    # ## Observations
    # - The above two graphs show the distribution of rainfall over months.
    # - The graphs clearly shows that amount of rainfall in high in the months july, aug, sep which is monsoon season in India.

    # In[8]:

    # data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
    #        'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").mean().plot.barh(stacked=True,figsize=(13,8));


    # In[9]:

    # data[['SUBDIVISION', 'JF', 'MAM',
    #        'JJAS', 'OND']].groupby("SUBDIVISION").sum().plot.barh(stacked=True,figsize=(16,8));


    # In[10]:

    # plt.figure(figsize=(11,4))
    # sns.heatmap(data[['JF', 'MAM',
    #        'JJAS', 'OND','ANNUAL']].corr(),annot=True)
    # plt.show()


    # In[11]:

    # plt.figure(figsize=(11,4))
    # sns.heatmap(data[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].corr(),annot=True)
    # plt.show()


    # ## Observations
    # - **Heat Map** shows the co-relation(dependency) betwenn the amounts of rainfall over months.
    # - From above it is clear that if amount of rainfall is high in the months of july, august, september then the amount of rainfall will be high annually.
    # - It is also obwserved that if amount of rainfall in good in the months of october, november, december then the rainfall is going to b good in the overall year. 

    # In[17]:

    #Function to plot the graphs
    def plot_graphs(groundtruth,prediction,title):        
        N = 9
        ind = np.arange(N)  # the x locations for the groups
        width = 0.27       # the width of the bars

        fig = plt.figure(figsize=(18,10))
        fig.suptitle(title, fontsize=12)
        ax = fig.add_subplot(111)
        rects1 = ax.bar(ind, groundtruth, width, color='m')
        rects2 = ax.bar(ind+width, prediction, width, color='c')

        ax.set_ylabel("Amount of rainfall")
        ax.set_xticks(ind+width)
        ax.set_xticklabels( ('APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC') )
        ax.legend( (rects1[0], rects2[0]), ('Ground truth', 'Prediction') )

    #     autolabel(rects1)
        for rect in rects1:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                    ha='center', va='bottom')
        for rect in rects2:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                    ha='center', va='bottom')
    #     autolabel(rects2)

        #plt.show()
        #mpld3.save_html(fig,'static/img/rainfall.html')
        plt.savefig('static/img/rainfall.png')


    # In[18]:

    def data_generation(year,region):
        temp = data[['SUBDIVISION','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
           'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == year]
        data_year = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
           'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == region])
        X_year = None; y_year = None
        for i in range(data_year.shape[1]-3):
            if X_year is None:
                X_year = data_year[:, i:i+3]
                y_year = data_year[:, i+3]
            else:
                X_year = np.concatenate((X_year, data_year[:, i:i+3]), axis=0)
                y_year = np.concatenate((y_year, data_year[:, i+3]), axis=0)
        
        return X_year,y_year


    # In[19]:

    def data_generation2(region):    
        Kerala = np.asarray(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
               'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['SUBDIVISION'] == region])

        X = None; y = None
        for i in range(Kerala.shape[1]-3):
            if X is None:
                X = Kerala[:, i:i+3]
                y = Kerala[:, i+3]
            else:
                X = np.concatenate((X, Kerala[:, i:i+3]), axis=0)
                y = np.concatenate((y, Kerala[:, i+3]), axis=0)

          
        return X,y


    # In[20]:

    def prediction2(year,region):
        from keras.models import Model
        from keras.layers import Dense, Input, Conv1D, Flatten

        # NN model
        inputs = Input(shape=(3,1))
        x = Conv1D(64, 2, padding='same', activation='elu')(inputs)
        x = Conv1D(128, 2, padding='same', activation='elu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='elu')(x)
        x = Dense(64, activation='elu')(x)
        x = Dense(32, activation='elu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=[inputs], outputs=[x])
        model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mae'])
        X_testing,Y_testing = data_generation(year,region)
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import explained_variance_score
        # linear model
        
        X_train,y_train = data_generation2(region)
        model.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=20, verbose=1, validation_split=0.1, shuffle=True)

        y_pred = model.predict(np.expand_dims(X_testing, axis=2))
        mae=mean_absolute_error(Y_testing, y_pred)
        score=explained_variance_score(Y_testing, y_pred)
        print(mae)
        print(score)
        
        Y_year_pred=list(range(9))
        for i in range(9):
            Y_year_pred[i]=y_pred[i][0]
        y_pred=np.array(Y_year_pred)
        plot_graphs(Y_testing,y_pred,"Year: "+str(year)+'  Region: '+str(region))
        return mae,score


    # In[ ]:




    # In[21]:
    print("############",year,type(int(year)),region,type(region),"77777777777777777777777777&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    mae,score=prediction2(int(year),region)
    mae=format(round(float(mae),2))
    score=format(round(float(score),2))
    keras.backend.clear_session()
    return mae,score



