import urllib
import sys
import cv2
#from win32api import GetSystemMetrics
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL.Image
import matplotlib.pyplot as plt
from scipy import ndimage, misc



def alignImages(im1, im2):
    #########under construction#########
  # Convert images to grayscale

  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.

  orb = cv2.ORB_create(MAX_FEATURES)

  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.

  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score

  matches = sorted(matches, key=lambda x: x.distance, reverse=False)

  # Remove not so good matches

  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches

  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches

  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

 

  for i, match in enumerate(matches):

    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt


  # Find homography

  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
   

  return im1Reg

filename = sys.argv[1]

#the [x, y] for each right-click event will be stored here
right_clicks = list()

#this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):

    #right-click event value is 2
    if event == 2:
        global right_clicks

        #store the coordinates of the right-click event
        right_clicks.append([x, y])
        print(right_clicks)
        

img = cv2.imread(filename,0)

#resizing the img
scale_width = 640 / img.shape[1]
scale_height = 480 / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', window_width, window_height)

#set mouse callback function for window

cv2.setMouseCallback('image', mouse_callback)

print("select 4 point with the right click and then hit <c>")
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", img)
	key = cv2.waitKey(1) & 0xFF

	# if the 'c' key is pressed, break from the loop
	if key == ord("c"):
		break

cv2.destroyAllWindows()

max_X = 0
max_Y = 0
min_X = right_clicks[0][0]
min_Y = right_clicks[0][1]

#finding cordintates from the image
for i in range (len(right_clicks)):
    
    if max_X < right_clicks[i][0]:
        max_X = right_clicks[i][0]
    
    if min_X > right_clicks[i][0]:
        min_X = right_clicks[i][0]

    if max_Y < right_clicks[i][1]:
        max_Y = right_clicks[i][1]
    
    if min_Y > right_clicks[i][1]:
        min_Y = right_clicks[i][1]
    

#setting new height,width
width = max_X - min_X
height = max_Y - min_Y
print(width)
print(height)

newImage = np.zeros((height,width))

print("min X-Y")
print(min_X)
print(min_Y)


print("max X-Y")
print(max_X)
print(max_Y)

a = 0
b = 0

#taking the pixel values that are in the window boundries and setting them in a new np.array
for i in range (max_Y,min_Y,-1):

    for j in range (min_X,max_X):

        newImage[a][b] = img[i][j]
        b = b + 1
    a = a + 1
    b = 0

#rotating the image 180 because was upside down
img_180 = ndimage.rotate(newImage, 180, reshape=False)

plt.imshow(img_180,cmap="gray")

plt.show()

Image.fromarray(img_180.astype(np.uint8)).save("output.png")
print("ok")

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15




# Read reference image
refFilename = "granma.jpg"

print("Reading reference image : ", refFilename)

imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)



# Read image to be aligned

imFilename = "output.png"

print("Reading image to align : ", imFilename); 

im = cv2.imread(imFilename, cv2.IMREAD_COLOR)


#########under construction #########

#imReg = alignImages(imReference, im)

# Write aligned image to disk.

#outFilename = "aligned.jpg"

#print("Saving aligned image : ", outFilename)

#cv2.imwrite(outFilename, imReg)

#plt.imshow(imReg,cmap="gray")

#plt.show()