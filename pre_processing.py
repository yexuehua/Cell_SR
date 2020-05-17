import matplotlib.pyplot as plt
import cv2
import os
files_name = os.listdir("./data")
files_path = [os.path.join("data",i) for i in files_name]
img = cv2.imread(files_path[0])
img_b,img_g,img_r = cv2.split(img)
print(type(img_b))
cv2.imshow("a",img_b)
cv2.waitKey(0)
cv2.destroyAllWindows()