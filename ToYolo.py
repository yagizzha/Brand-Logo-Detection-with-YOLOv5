
import cv2
import numpy as np
import os.path
from os import path

file1 = open('flickr_logos_27_dataset_training_set_annotation.txt', 'r')
Lines = file1.readlines()
base="flickr_logos_27_dataset_images/"
tobase="smallset/"
count = 0
# Strips the newline character
i=0
classes=[]
i=0
for line in Lines:
    postsplit=line.split()
    #print(postsplit)
    if not postsplit[1] in classes and len(classes)<10:
        classes.append(postsplit[1])
    if postsplit[1] in classes:
        filename=base+postsplit[0]
        filenameto=tobase+postsplit[0]
        current=cv2.imread(filename)
        y=len(current)
        x=len(current[0])
        inp=""
        for j in range(3,7):
            postsplit[j]=int(postsplit[j])
        #print("x=",x,"y=",y,"x1",postsplit[3],"x2",postsplit[5],"y1",postsplit[4],"y2",postsplit[6])
        inp+=str(classes.index(postsplit[1]))
        inp+=" "
        inp+=str("{:.6f}".format(((postsplit[3]+postsplit[5])/2)/x))
        inp+=" "
        inp+=str("{:.6f}".format(((postsplit[4]+postsplit[6])/2)/y))
        inp+=" "
        inp+=str("{:.6f}".format((postsplit[5]-postsplit[3])/x))
        inp+=" "
        inp+=str("{:.6f}".format((postsplit[6]-postsplit[4])/y))
        inp+="\n"
        cv2.imwrite(filenameto,current)
        x = filenameto.replace(".jpg", ".txt")
        print(not path.exists(filename))
        if not path.exists(filename):
            i+=1
        f=open(x, "w")
        f.write(inp)
        f.close()
    #print(inp)

print(i,"Files processed")
f=open(tobase+"classes.txt", "w")
for i in classes:
    f.write(i+"\n")
f.close()
