import cv2


def toPolarCoordinate(contour):
    '''
    Convert to polar coordinate using image moments
    '''
    pass


def test():
    img_file = "/home/data/duke_liver/dataset/mask/arterial/lrml_0143_ct/0025.png"
    img = cv2.imread(img_file,0)
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(img.shape)


def main():
    test()


if __name__ == "__main__":
    main()