import json
import os
import math

def main():
    filenames = dict()
    for filename in sorted(os.listdir('data/json_output/')):
        if filename.endswith(".json"):
            with open('data/json_output/' + filename) as json_data:
                d = json.load(json_data)
                if len(d['people']) > 0:
                    if filename[:9] not in filenames:
                        if filename[8].isalpha():
                            filenames[filename[:12]] = []
                        else:
                            filenames[filename[:9]] = []
                    if filename[8].isalpha():
                        filenames[filename[:12]].append\
                        (d['people'][0]['pose_keypoints_2d'][:3])
                    else:
                        filenames[filename[:9]].append\
                        (d['people'][0]['pose_keypoints_2d'][:3])

                    #print("Filename: {} {}".format(filename, d['people'][0]['pose_keypoints_2d'][3:6]))
            json_data.close
    i = 0
    for key, value in sorted(filenames.items()):
        movement = 0
        if len(value) > 1:
            for i,v in enumerate(value[1:]):
                movement += math.hypot(v[0] - value[i-1][0], v[1] - value[i-1][0] )
        movement = movement / len(value)
        stat = "{}, {}\n".format(key, movement)
        with open("movement_nose.csv", "a") as gz:
            gz.write(stat)


if __name__=="__main__":
    main()
