import json


dic={}
song={}
arr = []
index = 0
mapping={}

fullpath = "/mnt/data/recsys_spotify/submission/challenge_set.json"

f = open(fullpath)
js = f.read()
f.close()
slice = json.loads(js)

f = open("challenge",'w')




for line in slice["playlists"]:

	try:
		f.write(line['name'].lower()) 
	except KeyError:
		pass
	
	f.write('\n')
	
	for data in line['tracks']:
		f.write(data["track_uri"])
		f.write(' ')
	f.write('\n')
	
