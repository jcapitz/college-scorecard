import pandas as pd
file = '/Users/truong/Desktop/US_ED/'
year = range(1996,2014)
# schools = ['UNIVERSITY OF CALIFORNIA-IRVINE','UNIVERSITY OF CALIFORNIA-LOS ANGELES','UNIVERSITY OF CALIFORNIA-DAVIS','UNIVERSITY OF CALIFORNIA-BERKELEY','UNIVERSITY OF CALIFORNIA-MERCED','UNIVERSITY OF CALIFORNIA-RIVERSIDE','UNIVERSITY OF CALIFORNIA-SAN DIEGO','UNIVERSITY OF CALIFORNIA-SAN FRANCISCO','UNIVERSITY OF CALIFORNIA-SANTA BARBARA','UNIVERSITY OF CALIFORNIA-SANTA CRUZ']
schools = ['CHAPMAN UNIVERSITY']
uc_df = pd.DataFrame()

for i in range(0,len(year)):
	print(year[i])
	df = pd.read_csv(file + 'MERGED' + str(year[i]) + '_PP.csv',low_memory = False)
	condition = df['INSTNM'].map(lambda x: x in schools)
	uc_df = pd.concat([uc_df,df[condition]])

uc_df = uc_df.reset_index()
uc_df = uc_df.sort_values('INSTNM')




