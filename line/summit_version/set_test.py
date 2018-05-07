import json

song={}
mapping={}

disjoint_set={}


def find_set(x):
	try:
		if(disjoint_set[x]==x):
			return x
		else:
			disjoint_set[x] = find_set(disjoint_set[x])
			return disjoint_set[x]
	
	except KeyError:
		disjoint_set[x] = x
		return x

def joint(x,y):
	x = find_set(x)
	y = find_set(y)

	disjoint_set[y] = x


for i in range(1000):
	#print(i)
	fullpath = "/mnt/data/recsys_spotify/data/mpd.slice.{0}-{1}.json".format(i*1000,i*1000+999)

	f = open(fullpath)
	js = f.read()
	f.close()
	slice = json.loads(js)

	for line in slice["playlists"]:
		
		base = line['tracks'][0]
		try:
			song[ base["track_uri"] ] += 1
		except KeyError:
			song[ base["track_uri"] ] = 1
		

		for data in line['tracks'][1:]:
			
			joint(base["track_uri"],data["track_uri"])
            
			try:
				song[ data["track_uri"] ] += 1
			except KeyError:
				song[ data["track_uri"] ] = 1
        

fullpath = "/mnt/data/recsys_spotify/submission/challenge_set.json"
f = open(fullpath)
js = f.read()
f.close()
slice = json.loads(js)

for line in slice["playlists"]:
	if(len(line['tracks'])==0):
		continue

	base = line['tracks'][0]
	try:
		song[ base["track_uri"] ] += 1
	except KeyError:
		song[ base["track_uri"] ] = 1
	
	for data in line['tracks'][1:]:
		
		joint(base["track_uri"],data["track_uri"])
           
		try:
			song[ data["track_uri"] ] += 1
		except KeyError:
			song[ data["track_uri"] ] = 1
        






file_path = "/mnt/data/recsys_spotify/line_data/song_information/song_sort"
f = open(file_path,'w')
arr = [ (_,song[_]) for _ in song ]
arr = sorted(arr,key = lambda x:x[1] , reverse=True )

for line in arr:
	text = "{0}_{1}\n".format(line[0],line[1])
	f.write(text)
f.close()

index = 0
check_set = {}

for _ in disjoint_set:
	temp = find_set(_)
	try:
		check_set[temp].append(_)
	except KeyError:
		check_set[temp] = [_]
		index += 1

print(index)

i=0
for _ in check_set:
	file_path = "/mnt/data/recsys_spotify/line_data/song_information/song_set_{0}".format(i)
	f = open(file_path,'w')
	
	for line in check_set[_]:
		text=line+'\n'
		f.write(text)

	f.close()
	i += 1

