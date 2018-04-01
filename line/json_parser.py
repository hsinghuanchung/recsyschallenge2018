import json


dic={}
song={}
arr = []
index = 0
mapping={}

for i in range(800):
	#print(i)
	fullpath = "/mnt/data/recsys_spotify/data/mpd.slice.{0}-{1}.json".format(i*1000,i*1000+999)

	f = open(fullpath)
	js = f.read()
	f.close()
	slice = json.loads(js)

	for line in slice["playlists"]:

		check=0

		while(True):
			
			name = "{0}_{1}".format( line['name'] , check )

			if name in dic:
				check += 1
			else:
				dic[ name ] = []
				mapping[ name ] = index
				index += 1
				break


		for data in line['tracks']:
			dic[ name ].append(data["track_uri"])
            
			if data["track_uri"] in mapping:
				pass
			else:
				mapping[ data["track_uri"] ] = index
				index += 1

			try:
				song[ data["track_uri"] ] += 1
			except KeyError:
				song[ data["track_uri"] ] = 1
        


print(len(mapping))

f1 = open('rec_data','w')
f1.write(str(arr))
f2 = open('rec_data_2','w')
f2.write(str(arr))

f3 = open('mapping','w')
f3.write(str(len(mapping)))


ss = 0
for line in dic:
	for _ in dic[ line ]:
		s = "{0}\t{1}\n".format( mapping[line] , mapping[_]  )
		s1 = "{0}\t{1}\t{2}\n".format( mapping[line] , mapping[_] , 1  )
		f1.write(s)
		f2.write(s1)
		ss += 1
		
f3.write(str(ss))
for _ in mapping:
	s = '{0} {1}\n'.format(_,mapping[_])
	f3.write(s)

print(ss,"link")
print(len(mapping),"num")



