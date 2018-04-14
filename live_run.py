import cv2
from grab_screen import process_image

while True:
    image = process_image().flatten()
    image = image.reshape(23, 96)
    cv2.imshow('live', image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
