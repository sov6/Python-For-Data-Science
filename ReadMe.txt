
                                                                                    FACE DETECTION USING OPENCV
                                                                                    ----------------------------

Facial Detection is a computer technology capable of detecting human faces in digital image or a video frame from a video source. Face detection is being widely used in biometrics,
which is combined with facial recognition. With technology at its peak, it plays an important role in video surveillance, human computer interface and in image database management.
The project is implemented using python, computer vision and machine learning, where we train the system with thousands of images to detect faces.

The project provides the user with two choices and prompts the user to enter either 1 or 2. Choosing option 1 would let the user input a path to a location where the user would have
images stored, that can be used for face detection. We have implemented the face detection in images using both Haar and LBP classifier, thereby using the result to compare execution
time and accuracy of the two different cascade classifiers used.  Also,the input image used for testing will be displayed as output with detected faces in rectangle,for both the Haar
and LBP classifier. Choice 2 would enable the webcam of the user's laptop to capture video, which would be used as an input to our program. Here, both face and eyes of the user will
be detected and would be highlighted with rectangles. Users should be in a well-lit surroundings for the face detection to work.

GETTING STARTED:
----------------
The project is a low level implementation of a basic face detection functionality. To understand and to get the project working, following steps are recommended:

	•   A basic understanding of programming, python and importing libraries.
	•   A pre-installed python IDE where you can run the code, preferably Jupyter Notebook.
    	•   Access to a set of images to detect faces.
    	•   Access to the webcam to detect faces in video.
    
PREREQUISITES:
--------------
The programs utilises multiple python libraries for execution. Please have the following libraries for the code to run:

	•   Python Libraries:
            To work on the project, please get started by making the following libraries available in your python workspace:

	    	* OpenCV             		- (CV2)         - Library used to convert images from one format to another [BGR TO RGB, RGB TO GRAY], provides classifier object.
            	* Numpy              		- (numpy)       - Library used in scientific computing,includes 'n' dimensional array object,used in image processing with matplotlib.
            	* pyplot in matplotlib     	- (plt)         - Library used to  plot images and figures for output, matplotlib uses numpy to read and display images.
            	* time               		- (time)        - Library used to calculate the execution time of different types of classifiers


	•   Users downloading the program should also download XML files for Haar and LBP classifiers. We are detecting faces in images, face and eyes in video.
            Please download the following files:

             	* haarcascade_frontalface_default.xml                 - To detect face using Haar classifier
             	* haarcascade_eye.xml                                 - To detect eyes using Haar classifier
             	* lbpcascade_frontalface.xml                          - To detect face using LBP classifier

            You can download the files from the following links:

             	* https://github.com/parulnith/Face-Detection-in-Python-using-OpenCV/tree/master/data/haarcascades                 - For Haar classifiers
             	* https://github.com/opencv/opencv/tree/master/data/lbpcascades                                                    - For LBP classifiers

    	•   The function in the code selectClassifier(classifierType) returns classifier object based on classifierType argument. The function should be modified with the path 
    	    of user's directory/folder, where the Haar or LBP xml files saved.

    	•   Execution of the code would prompt the user to provide a path to a directory/folder where the user should store images for testing.

    	•   Users are encouraged to read more about the functions used from the python libraries mentioned above. This is to have a better understanding of the project
            and its functions.

INSTALLATION:
-------------

	•   INSTALLING JUPYTER NOTEBOOK:
            Python is a prerequisite for installation of Jupyter Notebook. Anaconda distribution can be used for installation of Python, Jupyter Notebook, and other packages used 
	    for scientific computing and data science. You can download and install anaconda from the following link:

            	* https://www.anaconda.com/distribution/

             Follow the instructions on the download page to complete the installation process. The page has instructions for Windows, MAC and Linux users.
        
    	     You can invoke Jupyter notebook from the command line with the command

            	* Jupyter notebook

	•   INSTALLING OPENCV:
            Executing the below command in command line would install openCV:
            
	        * pip install opencv-python

VERSION:
--------

	•   Python - Version used 3.7
	•   OpenCV - Version used 4.1.1.26

AUTHORS:
--------
	•   Anuja Mitruka
 	•   Lipi Kiliyelathu Paul
	•   Soumya Vikraman

ACKNOWLEDGEMENT:
----------------
We take this opportunity to thank our professor, Mr. Alexander Iliev, for providing us with enough resources and study materials to implement the project.
We would also like to express our gratitude to the following open source forums, where we could refer and get solutions to our technical queries.
	•Stack Overflow
	•Medium
	•Wikipedia
	•Python Documentation
	•OpenCV Documentation
    	•GeeksforGeeks    
    	•Google
 





