import cv2
import os
import re


images_path = './video3'
output_file = 'video3_saida.mp4'


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24

image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

pattern = r'\d+'

# Função para extrair o número inteiro do nome do arquivo
def extract_number(image_files):
    return int(re.findall(pattern, image_files)[0])

image_files = sorted(image_files, key=extract_number)


img = cv2.imread(os.path.join(images_path, image_files[0]))
height, width, channels = img.shape

out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

for image_file in image_files:
    img = cv2.imread(os.path.join(images_path, image_file))
    out.write(img)

out.release()
cv2.destroyAllWindows()
