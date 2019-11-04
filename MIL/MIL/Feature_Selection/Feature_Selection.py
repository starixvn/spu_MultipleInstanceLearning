import numpy as np
from pandas import read_csv
import os
from window_slider import Slider
import csv
import datetime as dt
from dateutil import parser
from itertools import tee
import bisect
from collections import Counter
import Transaction 
import Session 

def get_OS(ua):
    user_agent = parse(ua)
    return user_agent.os.family

def get_Browser(ua):
    user_agent = parse(ua)
    return user_agent.browser.family

def remove_space(code):
    return code.strip()

def convert(datetime_str): 
    datetime_object = parser.parse(datetime_str)
    return datetime_object 

def calTime(list):
    start=convert(list[0].TransDate)
    end=convert(list[1].TransDate)
    elapsedTime = end - start
    return elapsedTime.total_seconds()
 
def findIntersection(intervals,N): 
    l = intervals[0][0] 
    r = intervals[0][1] 
    for i in range(1,N): 
       if (intervals[i][0] > r or intervals[i][1] < l): 
           return-1 
       else: 
          l = max(l, intervals[i][0]) 
          r = min(r, intervals[i][1]) 
          
    return l,r

def removeDuplicate(listTrans):
    result=[]
    ids=[]
    for tupTrans in listTrans:
        for tran in tupTrans:
            if tran.TransID not in ids:
                ids.append(tran.TransID)
                result.append(tran)
    return result

def getIntervalPattern(listInterval,listTrans):
    stackNoneInterval=[]
    stackTrans=[]
    stackInterval=[]
    stackSession=[]
    stackNoneIntervalSession=[]
    stackNoneIntervalTrans=[]
    stackNoneIntervalIndex=[]
    for index in range(0,len(listInterval)):
      if len(stackInterval)==0:
          stackInterval.append(listInterval[index])
          continue
      num1 = stackInterval.pop()
      if listInterval[index] == num1:
          stackTrans.append(listTrans[index-1])
          stackTrans.append(listTrans[index])
          stackInterval.append(listInterval[index])
      else:
          stackNoneIntervalTrans.append(listTrans[index-1])
          stackNoneIntervalTrans.append(listTrans[index])
          stackNoneInterval.append(listInterval[index])
          stackNoneIntervalIndex.append(index)

    if len(stackTrans) > 10:
        stackTrans.extend(stackNoneIntervalTrans) 
        stackNoneIntervalTrans=[]
    else:
        stackNoneIntervalTrans.extend(stackTrans)
        stackTrans=[]

    stackSession.append(removeDuplicate(stackTrans))  
    stackNoneIntervalSession.append(removeDuplicate(stackNoneIntervalTrans))
    return stackSession,stackNoneIntervalSession
   


def IsHappenOccurrence(listInterval):
   num0 =  listInterval[0] 
   countOccurence = 0 
   for index in range(1,len(listInterval)):
        num1 = listInterval[index] 
        if num0 == num1:
            countOccurence =  countOccurence + 1
            num0 = num1
        else:
            num0 = num1
   
   if countOccurence > 3 :
       return True
   return False

 
def NumbersWithinRange(items, lower, upper):
    start = items.bisect(lower)
    end = items.bisect_right(upper)
    return items[start:end]

def buildRow(accountID,listTrans,day,status='False',fre=0,label='Human',key='hour'):
    rows=[]
    if len(listTrans) > 0 :
        start=convert(listTrans[0].TransDate)
        rows.append([accountID,listTrans[0].TransID,int(listTrans[0].DeviceID),listTrans[0].Year,listTrans[0].Month,listTrans[0].Day,listTrans[0].Hour,listTrans[0].Minute,listTrans[0].Second,listTrans[0].IP,listTrans[0].Browser,0,label])
        for index in range(1,len(listTrans)):
           end=convert(listTrans[index].TransDate)
           elapsedTime = end - start
           elapsedTime = round(elapsedTime.total_seconds()/60)
           rows.append([accountID,listTrans[index].TransID,int(listTrans[index].DeviceID),listTrans[index].Year,listTrans[index].Month,listTrans[index].Day,listTrans[index].Hour,listTrans[index].Minute,listTrans[index].Second,listTrans[index].IP,listTrans[index].Browser,elapsedTime,label])
           start = end
    return rows
   

def PatternInInterval(group,day='',key='hour'):
   listTrans=[]
   for item in group.values:
      obj= Transaction.Transaction(item)
      listTrans.append(obj)
    
   hasOccurrence = 0
   bucket_size = 2
   overlap_count = 1
   totalSeconds = 0
   seconds = 0
   session_robot=[]
   session_human=[]
   interval = []
   trans = []
   rowRobot=[]
   rowUnKnow=[]
   c_robot=0
   c_unknow=0
   if len(group) > 2 :
        listTrans = np.asarray(listTrans)
        slider = Slider(bucket_size,overlap_count)
        slider.fit(listTrans) 
        while True:
            window_data = slider.slide()
            if len(window_data) > 1:
                seconds = calTime(window_data)
                totalSeconds +=seconds
                interval.append(round(seconds/60))
                trans.append(window_data)
            if slider.reached_end_of_list(): break
        
        hasOccurrence = IsHappenOccurrence(interval)
   else:
        session_human.append(listTrans)

      
   if len(listTrans)> 0:
        if hasOccurrence:
            if key=='hour'  :
                rowRobot = buildRow(group['AccountID'].values[0],listTrans,convert(group['TransDate'].values[0]).strftime('%Y-%m-%d'),status='True',fre=0,label='1',key='hour')
            else:
                rowRobot = buildRow(group['AccountID'].values[0],listTrans,'',status='True',fre=0,label='1',key='day')
        else:
            if key=='hour':
                  rowUnKnow = buildRow(group['AccountID'].values[0],listTrans,convert(group['TransDate'].values[0]).strftime('%Y-%m-%d'),status='False',fre=0,label='0',key='hour')
            else:
                  rowUnKnow = buildRow(group['AccountID'].values[0],listTrans,'',status='False',fre=0,label='0',key='day')
         
  
   if len(rowRobot)> 0:
        with open(file_name_robot, 'a',newline='') as csvFile:
            for row in rowRobot:
                writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(row)
        csvFile.close()

   if len(rowUnKnow)>0:     
        with open(file_name_unknow, 'a',newline='') as csvFile:
            for row in rowUnKnow:
                writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(row)
        csvFile.close()
      
   

#file_count="D:/csv/Count.csv"
file_name_unknow="D:/NewTrain/NewTrain_0.csv"
file_name_robot="D:/NewTrain/NewTrain_1.csv"
df = read_csv('D:/csv/Alpha_2019_Jun.csv')
grouped= df.groupby(['AccountID'])

for name,group in grouped:
   group=group.sort_values(['TransDate'])
   group['DeviceID']=group['DeviceID'].fillna(0)
   group['Browser']=group['Browser'].fillna('Unknow')
   group['date'] = group['TransDate'].map(lambda x: convert(x).strftime('%Y-%m-%d'))
   group['year'] = group['TransDate'].map(lambda x: convert(x).year)
   group['month'] = group['TransDate'].map(lambda x: convert(x).month)
   group['day'] = group['TransDate'].map(lambda x: convert(x).day)
   group['hour'] = group['TransDate'].map(lambda x: convert(x).hour)
   group['minute'] = group['TransDate'].map(lambda x: convert(x).minute)
   group['second'] = group['TransDate'].map(lambda x: convert(x).second)
   group['time'] = group['TransDate'].map(lambda x: convert(x).strftime('%H-%M-%S'))
   group_in_day= group.groupby('day')
   for name_day,group_day in group_in_day:
      PatternInInterval(group_day,day=name_day)

   group_by_day= group[group.duplicated(['time'])].groupby('time')
   for item in group_by_day:
       PatternInInterval(group.loc[group['time'] == item[0]],day='',key='day')
   
    