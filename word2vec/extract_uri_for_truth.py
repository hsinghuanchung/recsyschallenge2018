
"""

    Usage:
        python extract_uri_for_truth.py [output file name]                           [range]
        python extract_uri_for_truth.py ../../word2vec/trunk/truth_uri_800000_800002 800000-800002 
        

"""


import sys
import json
import codecs
import datetime

name_url = set()
cache = {}
output_file_name = []

def get_playlist_name_url(playlist):
    ff = codecs.open(output_file_name,'a','utf-8') 
    #ff.write(str(playlist['pid']))
    #ff.write(" ")
    ff.write(str(playlist['num_tracks']));
    ff.write(" ")
    
    for track in playlist['tracks']:
        tmp_uri = track['track_uri'].split(':')
        ff.write(tmp_uri[2])
        ff.write(" ")

    ff.close()
   

def show_playlist(pid):
    if pid >=0 and pid < 1000000:
        low = 1000 * int(pid / 1000)
        high = low + 999
        offset = pid - low
        #path = "../../mnt/data/recsys_spotify/data/mpd.slice." + str(low) + '-' + str(high) + ".json"
        path = "../data/mpd.slice." + str(low) + '-' + str(high) + ".json"
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
                ff = codecs.open(output_file_name,'a','utf-8')
                """if pid == (iend - 1):
                    ff.write("ENDFILE")
                else:
                    ff.write("NEXT ")
                ff.close()"""

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

