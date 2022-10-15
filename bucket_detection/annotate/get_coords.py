import cv2
def Capture_Event(event, x, y, flags, params):
    	# If the left mouse button is pressed
	if event == cv2.EVENT_LBUTTONDOWN:
		# Print the coordinate of the 
		# clicked point
		print(f"({x}, {y})")
if __name__=="__main__":
    	# Read the Image.
    cap = cv2.VideoCapture('/home/balaji/Documents/code/RSL/Fish/videos/2068016.mp4')
    while(cap.isOpened()):
   
        ret, frame = cap.read()


       
        # Show the Image
        if(frame in [i for i in range(45)]):
            cv2.imshow('image', frame)
            # Set the Mouse Callback function, and call
            # the Capture_Event function.
            cv2.setMouseCallback('image', Capture_Event)
    
        # Press any key to exit``
        else:
            break
        cv2.waitKey(0)
	# Destroy all the windows