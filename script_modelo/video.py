import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

#    """ Video Path """
#video_path = "./videos/video1_cvat.mp4"
video_path = "./videos/video1cut_saida.mp4"
#video_path = "./videos/video2_saida.mp4"
#video_path = "./videos/video3_saida.mp4"


#""" Load the model """

model_paths =[ #"models/wire_unet-vanilla-17-04-dataset-ok.h5",
     #       "models/mestrado-UNET-11-04-23.h5",
    #        "models/mestrado-UNET-28-11.h5",
            "models/my_model-13-04-23-mobilenet.h5",
    #        "models/wire_unet-100epocas.h5",
    #        "models/wire_unet-adaptada-18-04-dataset-ok.h5"
]





for model_path in model_paths:


    model = tf.keras.models.load_model(model_path)

    a = []
    for layer in model.layers:
        a.append(layer.get_output_at(0).get_shape().as_list())
    input_size = a[0][1]


    #     Output Path
    name = video_path.split("/")[-1]
    name = name.split(".")[-2]
    out_model = model_path.split("/")[-1] 
    out_model = out_model.split(".")[-2]
    output = str(name + '-'+ out_model)

    #""" Reading frames """
    vs = cv2.VideoCapture(video_path)
    _, frame = vs.read()
    H, W, _ = frame.shape
    vs.release()

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(output+'.avi', fourcc, 10, (W, H))

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            out.release()
            break

        original_image = frame
        x = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
        #original_image = x
        h, w = x.shape

        x = cv2.resize(x, (input_size, input_size))
        x = x/255.0
        x = x.astype(np.float32)

        x = np.expand_dims(x, axis=0)
        pred_mask = model.predict(x)[0]
        pred_mask2 = cv2.cvtColor(pred_mask,cv2.COLOR_GRAY2BGR)
        pred_mask = np.concatenate(
            [
                pred_mask,
                pred_mask,
                pred_mask
            ], axis=2)
        pred_mask = (pred_mask > 0.1) * 255
        pred_mask = pred_mask.astype(np.float32)
        pred_mask = cv2.resize(pred_mask, (w, h))
        pred_mask = pred_mask.astype(np.uint8)

        original_image = original_image.astype(np.uint8)

        alpha = 0.6
        original_image2 = cv2.addWeighted(pred_mask, alpha, original_image, 1-alpha, 0)
        idx += 1

        out.write(original_image2)
        #cv2.imshow("out",original_image2)
        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break

   # cv2.destroyWindow("preview")
    cap.release()
print("fim")