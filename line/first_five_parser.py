import json
import sys

if(len(sys.argv) == 3):
	start = int(sys.argv[1])
	eend =  int(sys.argv[2])
else:
	print("there should be two number mention start and end")
	sys.exit(0)

f_in = open('first_five.in','w')
f_out = open('first_five.out','w')


for i in range(start,eend):
	#print(i)
	fullpath = "/mnt/data/recsys_spotify/data/mpd.slice.{0}-{1}.json".format(i*1000,i*1000+999)

	f = open(fullpath)
	js = f.read()
	f.close()
	slice = json.loads(js)
	index = 0

	for line in slice["playlists"]:
		index = 0
		num = int(line["num_tracks"])-5
		f_out.write(str(num))
		f_out.write(" ")

		for data in line['tracks']:
			if(index < 5):
				f_in.write(data["track_uri"])
				f_in.write(" ")
			else:
				f_out.write(data["track_uri"])
				f_out.write(" ")
			index += 1
		f_in.write("\n")
		f_out.write("\n")


