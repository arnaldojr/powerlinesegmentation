import cv2
import numpy as np

cap = cv2.VideoCapture("video1-cvat.mp4")

cont = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    
 #   if cont % 5 == 0:
    
    cv2.imwrite("./video1/frame%d.jpg" % cont, frame)
        #cv2.imshow("original", frame)
    
    if cv2.waitKey(1) == 27: # ESC para fechar a janela
        break
    cont += 1

print("frames: ", cont)
cv2.destroyAllWindows()
cap.release()