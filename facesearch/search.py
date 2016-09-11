import fnmatch
import numpy as np
import cv2
import sys
import time
import os
import subprocess

class Search:
    def __init__(self, argv):
        self.cascPath = argv[0]
        self.cap = cv2.VideoCapture(0)
        self.missingPath = argv[1]

    def webcam(self):
        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(self.cascPath)


        while(True):

            ret, frame = self.cap.read() # capture frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            #print "Found {0} faces!".format(len(faces))
            im = frame
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                im = frame[y:y+h,x:x+h]
                fname ="fotos/%d.jpg"%(time.time())
                cv2.imwrite(fname,im)
                name,val = self.find(fname)
                if(val > 1):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,name,(x+w, y), font, 1,(255,255,255),2)
                    break


            # Display the resulting frame
            cv2.imshow('frame',frame)
            #cv2.imshow('frame',im)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Press q to scape of imaging
                break
            #time.sleep(3)

        self.cap.release()
        cv2.destroyAllWindows()

    def find(self,fieldImg):
        results = {};
        maxval = -float("inf")
        match = ""
        for dirname, subdirs, fnames in os.walk( os.path.abspath( self.missingPath ) ) :
            for person in subdirs:
                personpath = os.path.join( dirname, person)
                scores = []
                for d,s,imgs in os.walk( os.path.join( dirname, person) ):
                    for img in imgs:
                        if fnmatch.fnmatch( img, '*.jpg' ):
                            imgpath = os.path.join(personpath,img)
                            cmd = ["br","-algorithm", "FaceRecognition", "-compare", imgpath, fieldImg]
                            #print cmd
                            #subprocess.call(cmd, shell=False)
                            p =subprocess.Popen( cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            out, err = p.communicate()
                            #print out
                            scores.append(float(out))
                #print scores
                average = reduce(lambda x, y: x + y, scores) / len(scores)
                #print  person, average
                results[person] = average
                if(average > maxval):
                    maxval = average
                    match = person
        print match, maxval
        return(match,maxval)

if __name__ == "__main__":
    s = Search(sys.argv[1:])
    s.webcam()
    target = sys.argv[3]
    #s.find(target)
