import discharge_prophet as dp
import flood_runoff_prophet as frp
import daily_runoff_prophet as drp
import weekly_runoff_prophet as wrp
import model 
import pandas as pd
import numpy as np
def fut_cal(user_date,end_date):
	if user_date > end_date :
		return 1 #Future Data
	else:
		return 0 #Existing Data

def drive(filename,user_date):
	#filename='godavari'
	data=pd.read_excel('data/'+filename+'.xlsx')

	user_date=pd.to_datetime(user_date)
	last_date=data['Date'].iloc[len(data)-1]
	last_date=pd.to_datetime(last_date)
	fut=fut_cal(user_date,last_date)


	#user_date='2019-01-06'
	#user_date='2005-01-06'
	
	check=0
	#fut=0(if existing date ) & 1(future date)
	#fut=1
	#-----if existing date is choosen--------------
	if fut==0 :
		
		for i in range(1,len(data.columns)):
			data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].mean())

		data['Date']=pd.to_datetime(data['Date'])

		def existing_preiction(i):
			discharge=data.Discharge[i]
			floodrunoff=data['flood runoff'][i]
			dailyrunoff=data['daily runoff'][i]
			weeklyrunoff=data['weekly runoff'][i]
			Flood=data.Flood[i]
			fd=[discharge,floodrunoff,dailyrunoff,weeklyrunoff]

			result,mae=model.flood_classifier(filename,fd)

			discharge=format(round(discharge,2))
			floodrunoff=format(round(floodrunoff,2))
			dailyrunoff=format(round(dailyrunoff,2))
			weeklyrunoff=format(round(weeklyrunoff,2))
			mae=format(round(mae,2))
			Predicted=''
			actual=''
			if result==0:
				predicted='Normal'
			else:
				predicted='High'
			if Flood==0:
				actual='Normal'
			else:
				actual='High'
			print("Features-")
			print("Discharge-",discharge,'floodrunoff-',floodrunoff,'dailyrunoff-',dailyrunoff,'weeklyrunoff-',weeklyrunoff)
			print("Actual-",Flood)
			print("Predicted-",result)
			print("Mean-Absolute-Error :",mae)
			results = {
					"discharge" : discharge,
					"floodrunoff" : floodrunoff,
					"dailyrunoff" : dailyrunoff,
					"weeklyrunoff" : weeklyrunoff,
					"meanabsoluteerrorr" : mae,
					"predicted" : predicted,
					"actualflood" : actual
				}
			# print("############",type(floodrunoff),type(mae),"#####################################")
			return results			


		for i in range(len(data.Date)):
		    if data.Date[i]==user_date :
		        results=existing_preiction(i)
		        check=1
		        return results
		        break;
		if check==0:
			print("choose valid Date")
			
	#----------if prediction is on future(upto two months)----------------------
	else :
		#wtd- 0-testing on 2 years , 1- predicting for future two months
		wtd=1
		d1=dp.discharge_forecast(filename,wtd)
		d2=frp.flood_runoff_forecast(filename,wtd)
		d3=drp.daily_runoff_forecast(filename,wtd)
		d4=wrp.weekly_runoff_forecast(filename,wtd)
		data1=pd.concat([d1,d2['flood runoff']],axis=1)
		data1=pd.concat([data1,d3['daily runoff']],axis=1)
		data1=pd.concat([data1,d4['weekly runoff']],axis=1)


		def future_preiction(i):
			discharge=data1.Discharge[i]
			floodrunoff=data1['flood runoff'][i]
			dailyrunoff=data1['daily runoff'][i]
			weeklyrunoff=data1['weekly runoff'][i]
			
			fd=[discharge,floodrunoff,dailyrunoff,weeklyrunoff]

			result,mae=model.flood_classifier(filename,fd)

			discharge=format(round(float(discharge),2))
			floodrunoff=format(round(float(floodrunoff),2))
			dailyrunoff=format(round(float(dailyrunoff),2))
			weeklyrunoff=format(round(float(weeklyrunoff),2))
			mae=format(round(mae,2))
			print("############floodrunoff-",type(floodrunoff)," Mae-",type(mae),"#####################################")

			print("Predicted Features-")
			print("Discharge-",discharge,'floodrunoff-',floodrunoff,'dailyrunoff-',dailyrunoff,'weeklyrunoff-',weeklyrunoff)
			print("Predicted-",result)
			print("Mean-Absolute-Error :",mae)
			Predicted=''
			actual=''
			if result==0:
				predicted='Normal'
			else:
				predicted='High'
			
			results = {
					"discharge" : discharge,
					"floodrunoff" : floodrunoff,
					"dailyrunoff" : dailyrunoff,
					"weeklyrunoff" : weeklyrunoff,
					"meanabsoluteerrorr" : 'NIL',
					"predicted" : predicted,
					"actualflood" : 'NIL'
				}
			return results			



		for i in range(len(data1.Date)):
		    if data1.Date[i]==user_date :
		        results=future_preiction(i)
		        check=1
		        return results
		        break;
		if check==0:
			print("choose valid future Date")
			