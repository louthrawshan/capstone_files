# capstone_files
## Custom.cpp and custom.yaml
Custom.cpp and custom.yaml are the files that I added to ORB2-SLAM code to make it work for the tank footage (https://github.com/raulmur/ORB_SLAM2)
custom.yaml is the calibration file for GoPro Hero 3 while the custom.cpp has been edited to accept video input for the localization algorithm

## feature_tracking.py
This is the standalone file that detects features using the FAST feature detector and subsequently tracks using KLT. The input file has to be a video
