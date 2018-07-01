
"""

    Usage:
        python extract_test_input.py [test filenme]     [output filename]                             
        python extract_test_input.py challenge_set.json ../../word2vec/trunk/task4_input
        python extract_test_input.py challenge_set.json ../../word2vec/trunk/task6_input
        python extract_test_input.py challenge_set.json ../../word2vec/trunk/task8_input
        python extract_test_input.py challenge_set.json ../../word2vec/trunk/task10_input
        python extract_test_input.py challenge_set.json ../../word2vec/trunk/task1_input
        python extract_test_input.py challenge_set.json ../task9_input_notitle
        python extract_test_input.py challenge_set.json ../task10_input_notitle
        python extract_test_input.py challenge_set.json ../task3_input
        python extract_test_input.py challenge_set.json ../task9_input_notitle_v2
        

"""

import re
import sys
import json
import datetime

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

if __name__ == '__main__':

    test_filename = sys.argv[1]

    with open(test_filename) as data_file:    
        data = json.load(data_file)

    output_filename = sys.argv[2]
    ff = open(output_filename, 'a')

    cnt = 0
    for i in range(len(data['playlists'])):
        
        if cnt == 1000:
            break;
          
        if data['playlists'][i]['num_samples'] == 100:
            """if cnt < 1000:
                cnt += 1
                continue
            """
            ff.write(str(data['playlists'][i]['pid']))
            ff.write(" ")

            """nname = normalize_name(data['playlists'][i]['name'])
            ff.write(nname.encode("utf-8"))
            ff.write(" ")
"""
            for track in data['playlists'][i]['tracks']:
                tmp_uri = track['track_uri'].split(':')
                ff.write(tmp_uri[2])
                ff.write(" ")
            cnt += 1
            
            #ff.write("NEXT ")
            

    ff.close()
    print "finish"



