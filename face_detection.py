##Anuja Mitruka, Lipi Kiliyelathu Paul, Soumya Vikraman
__author__ = "Anuja Mitruka, Lipi Kiliyelathu Paul, Soumya Vikraman"
__email__  = "anuja.mitruka@stud.srh-campus-berlin.de, lipikiliyelathupaul@stud.srh-campus-berlin.de, SOUMYA.VIKRAMAN@stud.srh-campus-berlin.de"
__build_python_version__ = "3.7"



import cv2                                                  #openCV library for face detection 
import numpy as np                                          #matplotlib dependancy 
import matplotlib.pyplot as plt                             #for plotting face detected images
import time                                                 #Haar/LBP execution time calculations
#output of plotting commands is displayed inline within frontends
%matplotlib inline                                          

def readImage(imagePath):                                   #reads image from specified path in BGR format
    return cv2.imread(imagePath)

def convertBGR2RGB(image):                                  #converts image from BGR to RGB
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def convertBGR2GRAY(image):                                 #converts image from RGB to Gray
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def selectClassifier(classifierType):                       #returns Haar face/Haar Eye/LBP classifier
    if classifierType == "Haar":
        return cv2.CascadeClassifier("/home/soumya/Desktop/SEMESTER_1/PyDataScience/Haar_Cascade_Files/haarcascade_frontalface_default.xml")
    elif classifierType == "LBP":
        return cv2.CascadeClassifier("/home/soumya/Desktop/SEMESTER_1/PyDataScience/LBP_Cascade_Files/lbpcascade_frontalface.xml") 
    elif classifierType == "HaarEye":
        return cv2.CascadeClassifier("/home/soumya/Desktop/SEMESTER_1/PyDataScience/Haar_Cascade_Files/haarcascade_eye.xml")
    else:
        print("We do not have the type of classifier that you provided!")
        return None

def drawRectangle(elementRects, image,isVideo,RGB = (0,255,0)): #draws rectangle on the faces in image/video 
    for (a,b,c,d) in elementRects:
        cv2.rectangle(image,(a,b),(a+c,b+d),RGB,5)              #draws rectangle on faces
        if isVideo:                                             #code executes only for video face detection
            colorFace = image[b:b+d,a:a+c]                      #extracts the image portion
            eyesDetected = elementDetect(colorFace,"HaarEye")   #calling elementDetect function to detect eyes
            

def elementDetect(image,classifierType):                        #detects the faces in the image
    scaleFactor = 1.2                                           #setting default values for scale factor             
    minNeighbors = 5                                            #setting default values for minNeighbors 
    isVideo = 0                                                 #working with image or camera
    elementClassifier = selectClassifier(classifierType)        #returns face classifier
    if elementClassifier != None:
        elementInImage = elementClassifier.detectMultiScale(image,scaleFactor,minNeighbors);  #returns rectangles on images
        drawRectangle(elementInImage,image,isVideo)                                           #draws rectangles on faces 
        return (len(elementInImage))                                                          #returns number of faces detected
    else:
        print("Improper classifier type, cannot be used for element detection")

        
def imagePlot(title,image):
    plt.figure()                                                #Plots a new figure
    plt.title(title)                                            #displays image Title
    plt.imshow(image)                                           #display rgb image with detected face on figure
    plt.show(block=False)                                       #prevents blocking of code execution after imshow is called
    plt.show()                                                  #Displays the figure 

def compareCascade(haarTime,lbpTime):                           #prints the faster classifier 
    print("\n")
    print("CONCLUSION".center(40))
    print("\n")
    if(haarTime > lbpTime):
        print("  LBP Classifier is faster than Haar, but less accurate")
    else:
        print("  Haar Classifier is faster than LBP")


flag = "YES"                                                #flag determines iteration of while loop
response = ['YES','Yes','yes','y','Y']                      #different types of inputs from user

while flag in response:                                     #entering while loop based on user input
    
    choice = int(input("Please enter your choice for inputing picture(1) or using webcam(2)"))
    
    if choice == 1:
                
        imgLocation = input("Enter the path where your test image is located. Path SHOULD include the image name.\n \n")             #takes image path from user

        bgrImg4test = readImage(imgLocation)               #reads image from location in BGR format
        
        rgbImg4test = convertBGR2RGB(bgrImg4test)          #converts BGR format to RGB format

        tick1 = time.time()                                #start time for Haar cascade facedetection execution
        haarFaceCount = elementDetect(rgbImg4test,"Haar")  #Haar cascade function
        tock1 = time.time()                                #end time for Haar cascade facedetection execution
        ticktockHaar = tock1-tick1
        print("\n Time of {} seconds was taken by the Haar Classifier for face detction \n" .format(ticktockHaar))
        
        haarTitle = "Face Detection Using Haar Classifier"
        imagePlot(haarTitle,rgbImg4test)                   #Plotting image with Haar cascade 
        
        tick2 = time.time()                                #start time for LBP cascade facedetection execution
        lbpFaceCount = elementDetect(rgbImg4test,"LBP")    #LBP cascade function
        tock2 = time.time()                                #end time for LBP cascade facedetection execution
        ticktockLBP = tock2-tick2
        print("\n Time of {} seconds was taken by the LBP Classifier for face detction \n" .format(ticktockLBP))
                
        lbpTitle = "Face Detection Using LBP Classifier"
        imagePlot(lbpTitle,rgbImg4test)                    #Plotting image with LBP cascade

        compareCascade(ticktockHaar,ticktockLBP)           #prints the faster classifier
        
    elif choice == 2:
        
        face_cascade = selectClassifier("Haar")            ##returns face classifier

        cap = cv2.VideoCapture(0)                          #Enables camera

        while True:                                                     
            ret, img = cap.read()                          #grabs next frame via camera, if successful,ret =True and returns the grabbed image
            faces = face_cascade.detectMultiScale(img)     #detects the face, returns rectangles on face
            drawRectangle(faces,img,1)                     #draws rectangle over face and eyes(isVideo set to 1)
            cv2.imshow('Video',img)                        #displays face and eye detected on video
            k = cv2.waitKey(1) & 0xff                      
            if k == 27:                                    #videocapture ends when user presses escape key 
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print(" \n Incorrect input. Please try again.")
        
    flag = input("\n Would you like to try other options\n")
