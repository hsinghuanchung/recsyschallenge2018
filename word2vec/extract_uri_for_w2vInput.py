"""

    Usage:
        
        python extract_uri_for_w2vInput.py [output file name] [range]
        python extract_uri_for_w2vInput.py train_uri_0_800000 0-800000

"""

import sys
import json
import codecs
import datetime
import re

cache = {}
output_file_name = []

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def get_playlist_name_url(playlist):
    ff = codecs.open(output_file_name,'a','utf-8') 

    #lower_name = playlist['name'].lower()
    nname = normalize_name(playlist['name'])
    ff.write(nname)
    ff.write(" ")

    a = 0
    for track in playlist['tracks']:
        tmp_uri = track['track_uri'].split(':')
        a = a + 1
        if a == 3:
            ff.write(nname)
            ff.write(" ")
            a = 1
        ff.write(tmp_uri[2])
        ff.write(" ")

    ff.write(nname)
    ff.write(" ")
    ff.close()
   

def show_playlist(pid):
    if pid >=0 and pid < 1000000:
        low = 1000 * int(pid / 1000)
        high = low + 999
        offset = pid - low
        path = "../../../../mnt/data/recsys_spotify/data/mpd.slice." + str(low) + '-' + str(high) + ".json"
        #path = "../data/mpd.slice." + str(low) + '-' + str(high) + ".json"
        if not path in cache:
            f = codecs.open(path, 'r', 'utf-8')
            js = f.read()
            f.close()
            playlist = json.loads(js)
            cache[path] = playlist

        playlist = cache[path]['playlists'][offset]
        get_playlist_name_url(playlist)

def show_playlists_in_range(start, end):
    try:
        istart = int(start)
        iend = int(end)
        if istart <= iend and istart >= 0 and iend <= 1000000:
            for pid in xrange(istart, iend):
                show_playlist(pid)
    except:
        raise
        print "bad pid"
    

if __name__ == '__main__':
    output_file_name = sys.argv[1]
    fields = sys.argv[2].split('-')
    if len(fields) == 2:
        start = fields[0]
        end = fields[1]
        show_playlists_in_range(start, end)
        
    print "finish"
