import cv2

# Load the image
image = cv2.imread("Data/train_crc/msi_0/mss001.jpg")

# Convert the image from RGB to BGR
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Display the image
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
