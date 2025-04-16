import tensorflow as tf
import cv2


def main():
    img = cv2.imread('mask.webp')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    print(img)
    print(img.shape)
    img = tf.nn.co

if __name__ == '__main__':
    main()