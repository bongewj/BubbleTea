# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 19:58:11 2021

@author: Eugene
"""
import numpy as np
import pandas as pd
import os
os.chdir(r'D:\DA stuff\Yale DV\BubbleTea')
os.getcwd()

#%%

import requests # library to handle requests

from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values
    
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

import folium # plotting library

#%%

# import demographics data
demo_chara = pd.read_excel('https://www.singstat.gov.sg/-/media/files/publications/ghs/ghs2015/excel/t1-9.xls',sheet_name='T7(Total)',header = 0)
demo_chara_c = demo_chara
demo_chara_c = demo_chara_c.dropna(how = 'all')
demo_chara_c = demo_chara_c.dropna(axis = 1, how = 'all')
demo_chara_c = demo_chara_c.drop(85, axis = 1)
demo_chara_c.columns = demo_chara_c.iloc[3]
demo_chara_c = demo_chara_c.iloc[5:]
subzones = demo_chara_c['Subzone'].dropna().unique()
subzones = list(filter(('Total').__ne__,subzones))
subzones = list(filter(('Peng Siang').__ne__,subzones))
subzones = list(filter(('Subzone').__ne__,subzones))
subzones = list(filter(('Central Subzone').__ne__,subzones))
subzones = list(filter(('Wenya').__ne__,subzones))
subzones = list(filter(('Istana Negara').__ne__,subzones))
subzones = list(filter(('Pasir Ris Wafer Fab Park').__ne__,subzones))
subzones = list(filter(('Plab').__ne__,subzones))
subzones = list(filter(('Gul Basin').__ne__,subzones))
subzones = list(filter(('Serangoon North Ind Estate').__ne__,subzones))
subzones = list(filter(('Tanjong Irau').__ne__,subzones))
subzones = list(filter(('Southern Group').__ne__,subzones))
subzones = list(filter(('Tuas Promenade').__ne__,subzones))
subzones = list(filter(('Tuas View Extension').__ne__,subzones))
subzones = list(filter(('Jurong Island And Bukom').__ne__,subzones))
subzones = ['Kebun Baru' if i=='Kebun Bahru' else i for i in subzones]
subzones = ['Bayfront' if i=='Bayfront Subzone' else i for i in subzones]
subzones = ['Ang Mo Kio Central' if i=='Ang Mo Kio Town Centre' else i for i in subzones]
subzones = ['National University of Singapore' if i=="National University Of S'pore" else i for i in subzones]
subzones = ['Sengkang Central' if i=='Sengkang Town Centre' else i for i in subzones]

#%%

planning_areas = demo_chara_c[~demo_chara_c['Subzone'].isnull()][['Planning Area','Subzone']]
planning_areas = planning_areas[planning_areas['Planning Area'] != 'Total']
planning_areas = planning_areas[planning_areas['Planning Area'] != 'Planning Area']

planning_areas['Subzone'] = ['Kebun Baru' if i=='Kebun Bahru' else i for i in planning_areas['Subzone']]
planning_areas['Subzone'] = ['Bayfront' if i=='Bayfront Subzone' else i for i in planning_areas['Subzone']]
planning_areas['Subzone'] = ['Ang Mo Kio Central' if i=='Ang Mo Kio Town Centre' else i for i in planning_areas['Subzone']]
planning_areas['Subzone'] = ['National University of Singapore' if i=="National University Of S'pore" else i for i in planning_areas['Subzone']]
planning_areas['Subzone'] = ['Sengkang Central' if i=='Sengkang Town Centre' else i for i in planning_areas['Subzone']]

planning_areas.reset_index(drop = True, inplace = True)

ind_pa = planning_areas[~planning_areas['Planning Area'].isnull()].index.tolist()

data = pd.DataFrame(columns = ['Planning Area','Subzone'])

for i in range(0,len(ind_pa)):
    
    if i < (len(ind_pa)-1):
        
        a = ind_pa[i]+1
        b = ind_pa[i+1]
        x = range(a,b,1)

        for j in x:
                    
            temp = [planning_areas['Planning Area'][ind_pa[i]],planning_areas['Subzone'][j]]
            temp = pd.Series(temp, index = data.columns)
            data = data.append(temp, ignore_index = True)
        
    if i == (len(ind_pa)-1):
        
        a = ind_pa[i]+1
        b = len(planning_areas)
        x = range(a,b,1)

        for j in x:
            
            temp = [planning_areas['Planning Area'][ind_pa[i]],planning_areas['Subzone'][j]]
            temp = pd.Series(temp, index = data.columns)
            data = data.append(temp, ignore_index = True)    

#%%

household_S = pd.read_excel('https://www.singstat.gov.sg/-/media/files/publications/ghs/ghs2015/excel/t148-152.xls',sheet_name='T151',header = 0)
# remove extra rows and columns
household_S = household_S.dropna(axis = 1, how = 'all')
household_S = household_S.dropna()
household_S.reset_index(inplace = True, drop = True)
household_S.columns = household_S.iloc[0]
household_S = household_S.iloc[2:]
household_S.rename(columns = {'Planning Area':'Neighborhood'}, inplace = True)
household_S.reset_index(inplace = True, drop = True)
towns = household_S['Neighborhood']
towns = list(filter(('Planning Area').__ne__,towns))
towns = list(filter(('Others').__ne__,towns))

#%% Singapore Map (coordinates of subzones)

lat_info = []
long_info = []

import time

for i in range(0,len(subzones)):
    
    location = subzones[i]
    address =  str(location) + ", Singapore"
    
    geolocator = Nominatim(user_agent="foursquare_agent")  
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    print(address,latitude, longitude)
    lat_info = np.append(lat_info,latitude)
    long_info = np.append(long_info,longitude)
    time.sleep(0.5)

#%%

lat = pd.DataFrame(lat_info)
long = pd.DataFrame(long_info)
names = pd.DataFrame(subzones)
sub_coord = pd.concat([names,lat,long],axis=1)
sub_coord.columns = ['Name','Lat','Long']
sub_coord = sub_coord.drop_duplicates(subset = ['Lat','Long'])

#%%

# Singapore Lat Long
address = "Singapore"
geolocator = Nominatim(user_agent="foursquare_agent")  
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude

map_singapore = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, neighborhood in zip(sub_coord['Lat'], sub_coord['Long'], sub_coord['Name']):
    label = folium.Popup(str(neighborhood), parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_singapore)  

map_singapore.save("mymap.html")

#%% extract coordinates of planning areas

lat_info2 = []
long_info2 = []

import time 

pa_coord = planning_areas[~planning_areas['Planning Area'].isnull()]['Planning Area']
pa_coord.reset_index(inplace = True, drop = True)

for i in range(0,len(pa_coord)):
    
    location = pa_coord[i]
    address =  str(location) + ", Singapore"
    
    geolocator = Nominatim(user_agent="foursquare_agent")  
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    print(address,latitude, longitude)
    lat_info2 = np.append(lat_info2,latitude)
    long_info2 = np.append(long_info2,longitude)
    time.sleep(0.5)

lat2 = pd.DataFrame(lat_info2)
long2 = pd.DataFrame(long_info2)
names2 = pd.DataFrame(pa_coord)
pa_coord = pd.concat([names2,lat2,long2],axis=1)
pa_coord.columns = ['Name','Lat','Long']
pa_coord = pa_coord.drop_duplicates(subset = ['Lat','Long'])

#%% Define Foursquare credentials

CLIENT_ID = '0OGT50X5XRDU5V5PAH0HYBFUXCMC1M4PYNUYTZ2PMJF3N4PK'
CLIENT_SECRET ='BLRVRECCNGXR02QZAQANBRJNXWAQX0GC034PVBWGYCSTGFPM'
VERSION = '20200501'
LIMIT = 200
QUERY = 'Bubble Tea'
print('Your credentials:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)

#radius = 50000
# url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, radius, LIMIT)
#url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&limit={}&query={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, LIMIT, QUERY)

import requests

#results = requests.get(url).json()

bbt_list=[]

for name, lat, lng in zip(temp['Name'], temp['Lat'], temp['Long']):
    
    print(name,lat,lng)
    
    # create the API request URL
    url = 'https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&limit={}&query={}'.format(CLIENT_ID, CLIENT_SECRET, lat,lng, VERSION, LIMIT, QUERY)
        
    # make the GET request
    results = requests.get(url).json()["response"]['groups'][0]['items']
    
#    if 'venues' in results:
#        results = results['venues']
#    else:
#        results = results
    
    # return only relevant information for each nearby venue
    bbt_list.append([(
        name, 
        lat, 
        lng, 
        v['venue']['name'], 
        v['venue']['location']['lat'], 
        v['venue']['location']['lng'],  
        v['venue']['location']['formattedAddress'][0]) for v in results])

#%%

expanded_bbt = pd.DataFrame(columns = ['Subzone','Lat','Long','Name','Lat2','Long2','City'])

for i in range(0,len(bbt_list)):
    
    temp_list = bbt_list[i]
    
    for j in range(0,len(temp_list)):
        
        temp_list2 = temp_list[j]      
        temp2 = pd.DataFrame(temp_list2)
        temp2 = temp2.T
        temp2.columns = ['Subzone','Lat','Long','Name','Lat2','Long2','City']
        expanded_bbt = expanded_bbt.append(temp2, ignore_index = True)

temp_name = [x.lower() for x in expanded_bbt['Name']]
expanded_bbt['Name_Lower'] = temp_name

temp_pa = [data[data['Subzone'] ==  x]['Planning Area'].values for x in expanded_bbt['Subzone']]

#%%

from haversine import haversine as hs

temp_pa = []

for x in range(len(expanded_bbt)):
    
    sub = (expanded_bbt.loc[x,['Lat2']][0],expanded_bbt.loc[x,['Long2']][0])
    
    temp_hav = []
    temp_loc = 0
    
    for k in range(len(pa_coord)):
        
        temp_coord = (pa_coord.loc[k,['Lat']][0],pa_coord.loc[k,['Long']][0])
        
        if not temp_hav:
            
            temp_hav = hs(sub,temp_coord)
            temp_loc = k
        
        else:
            
            temp_hav2 = min(temp_hav, hs(sub,temp_coord))
            
            if temp_hav2 != temp_hav:
                
                temp_loc = k
            
            temp_hav = temp_hav2
        
    print(expanded_bbt.loc[x,'City'],pa_coord.loc[temp_loc,'Name'])
    
    temp_pa.append(pa_coord.loc[temp_loc,'Name'])

expanded_bbt['Matched Planning Area'] = temp_pa

#%%

bbt_names2 = expanded_bbt['Name_Lower']

bbt_names2 = ['koi' if 'koi ' in i else i for i in bbt_names2]
bbt_names2 = ['chicha san chen' if  all(x in i for x in ['chi','cha','san','chen']) else i for i in bbt_names2]
bbt_names2 = ['each a cup' if  all(x in i for x in ['each','a','cup']) else i for i in bbt_names2]
bbt_names2 = ['eskimo cafe & dessert bar' if  all(x in i for x in ['eskimo','cafe','dessert','bar']) else i for i in bbt_names2]
bbt_names2 = ['gong cha' if  all(x in i for x in ['gong','cha']) else i for i in bbt_names2]
bbt_names2 = ['heytea' if  all(x in i for x in ['heytea']) else i for i in bbt_names2]
bbt_names2 = ['i ♥ taimei' if  all(x in i for x in ['i','tai','mei']) else i for i in bbt_names2]
bbt_names2 = ['i.tea' if  all(x in i for x in ['i-tea']) else i for i in bbt_names2]
bbt_names2 = ['i.tea' if  all(x in i for x in ['i','. tea']) else i for i in bbt_names2]
bbt_names2 = ['i.tea' if  all(x in i for x in ['itea']) else i for i in bbt_names2]
bbt_names2 = ['juz bread' if  all(x in i for x in ['juz','bread']) else i for i in bbt_names2]
bbt_names2 = ['liho' if  any(x in i for x in ['liho','li ho']) else i for i in bbt_names2]
bbt_names2 = ['milksha' if  all(x in i for x in ['milksha']) else i for i in bbt_names2]
bbt_names2 = ['man ting xiang tea house' if  all(x in i for x in ['ting','xiang','tea']) else i for i in bbt_names2]
bbt_names2 = ['partea' if  all(x in i for x in ['partea']) else i for i in bbt_names2]
bbt_names2 = ['playmade' if  all(x in i for x in ['playmade']) else i for i in bbt_names2]
bbt_names2 = ['r&b tea' if  all(x in i for x in ['r&b']) else i for i in bbt_names2]
bbt_names2 = ['share tea' if  all(x in i for x in ['share','tea']) else i for i in bbt_names2]
bbt_names2 = ['sweet talk' if  all(x in i for x in ['sweet','talk']) else i for i in bbt_names2]
bbt_names2 = ['tea valley' if  all(x in i for x in ['tea','valley']) else i for i in bbt_names2]
bbt_names2 = ['tealive' if  all(x in i for x in ['tealive']) else i for i in bbt_names2]
bbt_names2 = ["ten ren's tea" if  all(x in i for x in ['ten','ren','tea']) else i for i in bbt_names2]
bbt_names2 = ['the alley' if  all(x in i for x in ['the','alley']) else i for i in bbt_names2]
bbt_names2 = ['the whale tea' if  all(x in i for x in ['the','whale','tea']) else i for i in bbt_names2]
bbt_names2 = ['tiger sugar' if  all(x in i for x in ['tiger','sugar']) else i for i in bbt_names2]
bbt_names2 = ['xing fu tang' if  all(x in i for x in ['xing','fu','tang']) else i for i in bbt_names2]
bbt_names2 = ['yuan cha' if  all(x in i for x in ['yuan','cha','tang']) else i for i in bbt_names2]
bbt_names2 = ['一点点' if  all(x in i for x in ['一点点']) else i for i in bbt_names2]
bbt_names2 = ['一芳台湾水果茶' if  all(x in i for x in ['一芳']) else i for i in bbt_names2]

bbt_names3 = sorted(bbt_names2)

expanded_bbt.loc[:,'Name_Lower'] = bbt_names2
expanded_bbt_2 = expanded_bbt.drop_duplicates(['Name_Lower','Lat2','Long2','City'])
duplicatedrows = expanded_bbt[expanded_bbt.duplicated(['Name_Lower','Lat2','Long2','City'])]

#%%

#expanded_bbt_grp = expanded_bbt_2.groupby('Subzone')['Name_Lower'].apply(lambda x: x.value_counts().index[0]).reset_index()
#expanded_bbt_grp2 = expanded_bbt_2.groupby(['Subzone','Name_Lower']).agg(lambda x:x.value_counts().index[0])
expanded_bbt_grp = expanded_bbt_2.groupby(['Matched Planning Area','Name_Lower']).size().to_frame(name = 'Count').reset_index()
#expanded_bbt_grp2 = expanded_bbt_grp.groupby(['Matched Planning Area'], sort = False)['Count'].max()
idx = expanded_bbt_grp.groupby(['Matched Planning Area'])['Count'].transform(max) == expanded_bbt_grp['Count']
expanded_bbt_grp3 = expanded_bbt_grp[idx]

#%%

with_max = expanded_bbt_grp3.groupby(['Matched Planning Area']).size().to_frame(name = 'Count Stores').reset_index()
with_max = with_max[with_max['Count Stores'] == 1].reset_index(drop = True)

no_max = expanded_bbt_grp3.groupby(['Matched Planning Area']).size().to_frame(name = 'Count Stores').reset_index()
no_max = no_max[no_max['Count Stores'] != 1].reset_index(drop = True)

with_max = pd.merge(with_max,expanded_bbt_grp3,on = 'Matched Planning Area').drop(columns = ['Count Stores'])
with_max = pd.merge(with_max,pa_coord,left_on = 'Matched Planning Area',right_on = 'Name')
with_max = with_max[with_max['Matched Planning Area'] != 'Lim Chu Kang']
with_max = with_max[with_max['Matched Planning Area'] != 'Simpang']

no_max = pd.merge(no_max,expanded_bbt_grp3,on = 'Matched Planning Area').drop(columns = ['Count Stores'])
no_max_dropped = pd.merge(no_max,pa_coord,left_on = 'Matched Planning Area',right_on='Name')

#%%

no_max = no_max.drop_duplicates(['Matched Planning Area'])
#%%

color_keys = with_max['Name_Lower'].unique()
color_values = ['red', 'blue', 'green', 'purple', 'orange', 'darkred','lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
colors = color_values[0:len(color_keys)]
dictionary = dict(zip(color_keys,colors))

#%%

# Singapore Lat Long
address = "Singapore"
geolocator = Nominatim(user_agent="foursquare_agent")  
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude

map_singapore2 = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, neighborhood,count,name in zip(with_max['Lat'], with_max['Long'], with_max['Matched Planning Area'],with_max['Count'],with_max['Name_Lower']):
    label = folium.Popup(str(name), parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=count*3,
        popup=label,
        color=dictionary[name],
        fill=True,
        fill_color= dictionary[name],
        fill_opacity=0.7,
        parse_html=False).add_to(map_singapore2) 
    
for lat, lng, neighborhood,count in zip(no_max_dropped['Lat'], no_max_dropped['Long'], no_max_dropped['Matched Planning Area'],no_max_dropped['Count']):
    label = folium.Popup(str(neighborhood), parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=count*3,
        popup=label,
        color='gray',
        fill=True,
        fill_color='gray',
        fill_opacity=0.7,
        parse_html=False).add_to(map_singapore2)  

map_singapore2.save("mymap2.html")

#%%

import re

bbt_names3 = [re.sub("([^\x00-\x7F])+"," ",x) for x in bbt_names] # drop non english characters
bbt_names3 = [re.sub("[\(\[].*?[\)\]]", "", x) for x in bbt_names2] # remove brackets
#bbt_names2 = [x.replace(" ", "") for x in bbt_names2]

#%%

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import dbscan
from Levenshtein import distance

data = bbt_names2

def lev_metric(x,y):
    i, j = int(x[0]), int(y[0])     # extract indices
    return distance(data[i], data[j])

X = np.arange(len(data)).reshape(-1, 1)

db = dbscan(X, metric=lev_metric, eps = 2, min_samples=2, algorithm = 'brute')

temp1 = pd.DataFrame(data)
temp2 = pd.DataFrame(db[1])
tempdf = pd.concat([temp1,temp2], axis = 1)
tempdf.columns = ['Names','Cluster']

#%%

import numpy as np
from sklearn.cluster import AffinityPropagation
from Levenshtein import distance
    
words = bbt_names2 #Replace this line
words = np.asarray(words) #So that indexing with a list will work
lev_similarity = -1*np.array([[distance(w1,w2) for w1 in words] for w2 in words])

#affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
affprop = AffinityPropagation(damping=0.5)
affprop.fit(lev_similarity)
for cluster_id in np.unique(affprop.labels_):
    exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    cluster_str = ", ".join(cluster)
    print(" - *%s:* %s" % (exemplar, cluster_str))

temp1 = pd.DataFrame(words)
temp2 = pd.DataFrame(affprop.labels_)
tempdf2 = pd.concat([temp1,temp2], axis = 1)
tempdf2.columns = ['Names','Cluster']
