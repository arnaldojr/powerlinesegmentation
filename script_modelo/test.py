import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_sample(X,y , y_pred, blend_preds):
    """Function to plot the results"""

    
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    
    ax[0].imshow(cv2.cvtColor(X,cv2.COLOR_BGR2RGB))
    ax[0].set_title('Entrada')
    ax[0].axis("off")

    ax[1].imshow(y,cmap="gray")
    ax[1].set_title('Label')
    ax[1].axis("off")

    ax[2].imshow(y_pred)
    ax[2].set_title('Predição')
    ax[2].axis("off")

    ax[3].imshow(cv2.cvtColor(blend_preds,cv2.COLOR_BGR2RGB))
    ax[3].set_title('Sobreposição')
    ax[3].axis("off")

    return fig


if __name__ == "__main__":


    """ Load the test images """
    test_images = glob("images/test/*")
    test_label = glob("images/test_gt/*")

    kernel = np.ones((5,5),np.uint8)


    model_paths =[# "models/wire_unet-vanilla-17-04-dataset-ok.h5",
        #       "models/my_model-13-04-23.h5",
                "models/mestrado-UNET-11-04-23.h5"
        #        "models/mestrado-UNET-2023-marco.h5",
         #      "models/my_model-13-04-23-vggunet.h5"
        #        "models/wire_unet-adaptada-18-04-dataset-ok.h5"
    ]


    for model_path in model_paths:

        """ Load the model """

        model = tf.keras.models.load_model(model_path)

        a = []
        for layer in model.layers:
            a.append(layer.get_output_at(0).get_shape().as_list())
        input_size = a[0][1]

        #     Output Path
        out_model = model_path.split("/")[-1] 
        out_model = out_model.split(".")[-2]
        output = str(out_model)
        
        for path in tqdm(test_images, total=len(test_images)):
            original_image = cv2.imread(path)
            
            x = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
            #original_image = x
            h, w = x.shape

            x = cv2.resize(x, (input_size, input_size))
            x = x/255.0
            x = x.astype(np.float32)

            x = np.expand_dims(x, axis=0)
            pred_mask = model.predict(x)[0]
            pred_mask_orig = pred_mask
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

            # Encontre o índice correspondente do arquivo de rótulo
            base_name = os.path.basename(path)
            base_name_without_ext, _ = os.path.splitext(base_name)
            #print(base_name_without_ext)
            label_base_name = base_name_without_ext + ".png"  # Substitua a extensão por .png
            #print("nome corrigido:", label_base_name)
            label_path = os.path.join("images", "test_gt", label_base_name)
            #print("label_path", label_path)
           
            #label_index = test_label.index(label_path)
            img_label = cv2.imread(label_path)  # Use o índice encontrado
            y_label = cv2.cvtColor(img_label, cv2.COLOR_BGR2GRAY)
            y_label = cv2.morphologyEx(y_label, cv2.MORPH_CROSS, kernel)



            fig = plot_sample(original_image, y_label, pred_mask, original_image2)

            name = path.split("\\")[-1]
            fig.savefig(f"save_images/{output}-img_junto-{name}.png")
            plt.close(fig)

            cv2.imwrite(f"save_images/{output}-img-{name}", original_image2)
            cv2.imwrite(f"save_images/{output}-mask-{name}", pred_mask)
