import ast
import cv2
import os
import subprocess
import re
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

input_command = "F:\\fsoft.qai.ppocr.vietnamesePhase\\qai_test\\private_test_imgs\\img_23.jpg"
command = [
    'python',
    'PaddleOCR-Vietnamese/tools/infer/predict_det.py',
    '--det_algorithm=SAST',
    '--det_model_dir=F:\\paddleocrDemo\\det_db_inference',
    f'--image_dir={input_command}',
    '--use_gpu=False'
]
subprocess.run(command)

with open('inference_results\\det_results.txt', 'r') as file:
    content = file.read()

start_index = content.find('[')
trimmed_content = content[start_index:]
data_list = ast.literal_eval(trimmed_content)
image_path = input_command
image = cv2.imread(image_path)

image_name = os.path.splitext(os.path.basename(image_path))[0]

output_folder = f'output_images_{image_name}'
os.makedirs(output_folder, exist_ok=True)

for i, bounding_box in enumerate(data_list):
    bounding_box = [(int(point[0]), int(point[1])) for point in bounding_box]
    roi = image[bounding_box[0][1]:bounding_box[2]
                [1], bounding_box[0][0]:bounding_box[2][0]]
    umber_second = int(bounding_box[3][0])
    output_path = os.path.join(output_folder, f'output_{umber_second}.jpg')
    cv2.imwrite(output_path, roi)

rec_command = [
    'python',
    'PaddleOCR-Vietnamese/tools/infer/predict_rec.py',
    f'--image_dir={output_folder}',
    '--rec_algorithm=SRN',
    '--rec_model_dir=SRN',
    '--rec_image_shape=3, 64, 256',
    '--rec_char_dict_path=vi_vietnam.txt',
    '--use_gpu=False'
]

subprocess.run(rec_command)

file_path = 'output_predictions.txt'
output_file_path = 'sorted_output.txt'
image_output_path = 'output_image.png'

with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in lines:
        data = eval(line)
        sorted_data = sorted(data, key=lambda x: int(
            re.search(r'output_(\d+).jpg', x[0]).group(1)))
        for item in sorted_data:
            output_file.write(item[1][0] + ' ')
        output_file.write('\n')

image_size = (800, 600)
background_color = (255, 255, 255)
text_color = (0, 0, 0)
font_size = 20
font_path = "SVN-Arial-Regular.ttf"

image = Image.new('RGB', image_size, background_color)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype(font_path, font_size)

with open(output_file_path, 'r', encoding='utf-8') as output_file:
    lines = output_file.readlines()
    for i, line in enumerate(lines):
        draw.text((10, i * font_size), line.strip(), text_color, font=font)

image.save(image_output_path)
det_result_image = cv2.imread(f"inference_results\\det_res_{image_name}.jpg")
image_cv2 = cv2.imread(image_output_path)
cv2.imshow('Sorted Output Image', image_cv2)
cv2.imshow("Detected Results", det_result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
