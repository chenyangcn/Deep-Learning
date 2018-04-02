import cv2
import sys
import numpy as np
import tensorflow as tf
from model import predict, image_to_tensor, deepnn

modelPath = './models'
imageBasicPath = './pic/'
EMOTIONS = ['angry', 'disgusted', 'fearful',
            'happy', 'sad', 'surprised', 'neutral']


def work(imagePath):
    # Set up paraments
    face_x = tf.placeholder(tf.float32, [None, 2304])
    y_conv = deepnn(face_x)
    probs = tf.nn.softmax(y_conv)
    # Read model
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(modelPath)
    sess = tf.Session()
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore model sucsses!!')

    faceCascade = cv2.CascadeClassifier(
        "./data/haarcascade_files/haarcascade_frontalface_default.xml")  # Create the haar cascade
    image = cv2.imread(imagePath)  # Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Image grayly
    faces = faceCascade.detectMultiScale(  # Detect faces in the image
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(5, 5),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
        print(x, y, h, w)
        cv2.rectangle(image, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)  # Frame out the head

        # Predict
        crop = gray[y:y+h, x:x+w]
        crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_CUBIC)
        tensor = image_to_tensor(crop)
        result = sess.run(probs, feed_dict={face_x: tensor})
        print(result)
        if result is not None:
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(image, emotion, (10, index * 20 + 20),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.rectangle(image, (130, index * 20 + 10), (130 +
                                                            int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)
        emoji_max = EMOTIONS[np.argmax(result[0])] # Pick out emotion
        print(emoji_max)
        cv2.putText(image, emoji_max, (x+w, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Faces found", image)  # 7
    cv2.waitKey(0)  # 8


if __name__ == '__main__':
    imagePath = imageBasicPath + sys.argv[1]  # Get user supplied values
    work ( imagePath )
