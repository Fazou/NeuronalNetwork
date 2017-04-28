import cv2

imgg = cv2.imread("..//Data//2seancephoto//cylindrejaune//15.jpg", 1)

detector = cv2.ORB_create()

for attribute in dir(new_detector):
    if not attribute.startswith("get"):
        continue
    param = attribute.replace("get", "")
    get_param = getattr(new_backend, attribute)
    val = get_param()
    print param, '=', val