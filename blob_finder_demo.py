from __future__ import print_function
import cv2
import cv2.cv as cv
import numpy

class BlobFinder(object):

    def __init__(self, threshold=200, filterByArea=True, minArea=100, maxArea=None):
        self.threshold = threshold
        self.filterByArea = filterByArea 
        self.minArea = minArea 
        self.maxArea = maxArea 

    def find(self,image):

        rval, threshImage = cv2.threshold(image, self.threshold,255,cv.CV_THRESH_BINARY)
        contourList, dummy = cv2.findContours(threshImage, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)

        # Find blob data
        blobList = []
        blobContours = []

        for contour in contourList:

            blobOk = True

            # Get area and apply area filter  
            area = cv2.contourArea(contour)
            if self.filterByArea:
                if area <= 0:
                    blobOk = False
                if self.minArea is not None:
                    if area < self.minArea:
                        blobOk = False
                if self.maxArea is not None:
                    if area > self.maxArea:
                        blobOk = False

            # Get centroid
            moments = cv2.moments(contour)
            if moments['m00'] > 0 and blobOk:
                centroidX = moments['m10']/moments['m00']
                centroidY = moments['m01']/moments['m00']
            else:
                blobOk = False
                centroidX = 0.0
                centroidY = 0.0

            # Get bounding rectangle
            if blobOk:
                bound_rect = cv2.boundingRect(contour)
                minX = bound_rect[0]
                minY = bound_rect[1]
                maxX = bound_rect[0] + bound_rect[2] 
                maxY = bound_rect[1] + bound_rect[3] 
            else:
                minX = 0.0 
                minY = 0.0
                maxX = 0.0
                maxY = 0.0

            # Create blob dictionary
            blob = {
                    'centroidX' : centroidX,
                    'centroidY' : centroidY,
                    'minX'      : minX,
                    'maxX'      : maxX,
                    'minY'      : minY,
                    'maxY'      : maxY,
                    'area'      : area,
                    } 

            # If blob is OK add to list of blobs
            if blobOk: 
                blobList.append(blob)
                blobContours.append(contour)

        # Draw blob on image
        blobImage = cv2.cvtColor(image,cv.CV_GRAY2BGR)
        cv2.drawContours(blobImage,blobContours,-1,(0,0,255),3)

        return blobList, blobImage



# -----------------------------------------------------------------------------
if __name__ == '__main__':

    import os
    import time

    imageExt = '.bmp'
    imageDir = os.path.join(os.curdir, 'images')
    sleepDt = 300  # sleep time between images in ms - slow things down for viewing the demo

    # Blob finder parameters
    threshold = 120 
    filterByArea = True 
    minArea = 200 
    maxArea = None

    # Get list of image Files
    imageFileList = os.listdir(imageDir)
    imageFileList = [f for f in imageFileList if os.path.splitext(f)[1]==imageExt]

    # Preview windows
    cv2.namedWindow('rawImage',0)
    cv2.namedWindow('invImage',0)
    cv2.namedWindow('blobImage',0)

    blobFinder = BlobFinder(
            threshold=threshold,
            filterByArea=filterByArea,
            minArea=minArea,
            maxArea=maxArea
            )

    for imageFile in imageFileList:

        print('Processing: ', imageFile)

        # Load image from file - note bmp images are by default BGR - convert to mono
        imageFullPath = os.path.join(imageDir,imageFile)
        rawImage = cv2.imread(imageFullPath)
        rawImage = cv2.cvtColor(rawImage,cv.CV_BGR2GRAY)


        # Create inverse image - so flies show up white (assums 8bit images)
        invImage = 255 - rawImage

        # Find blobs
        blobList, blobImage = blobFinder.find(invImage)
        print('  found {0} blobs'.format(len(blobList)))

        for i, blob in enumerate(blobList):
            print('    blob #{0}'.format(i))
            for k,v in blob.iteritems():
                print('      {0}: {1}'.format(k,v))

        cv2.imshow('rawImage',rawImage) 
        cv2.imshow('invImage',invImage)
        cv2.imshow('blobImage',blobImage)


        cv2.waitKey(sleepDt)
        




