import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
import ast
import subprocess
import re
from PIL import Image, ImageDraw, ImageFont, ImageTk


class Inference:
    def __init__(self, app):
        self.app = app
        self.result_label = tk.Label(self.app.root, text="")
        self.result_label.pack()

    def infer(self, image_path):
        input_command = image_path
        command = [
            'python',
            'PaddleOCR-Vietnamese/tools/infer/predict_det.py',
            '--det_algorithm=SAST',
            '--det_model_dir=det_db_inference',
            f'--image_dir={input_command}',
            '--use_gpu=False'
        ]
        subprocess.run(command)
        # Đọc nội dung từ file txt
        with open('inference_results\\det_results.txt', 'r') as file:
            content = file.read()

        start_index = content.find('[')
        trimmed_content = content[start_index:]
        data_list = ast.literal_eval(trimmed_content)

        # Đọc ảnh
        image_path = input_command
        image = cv2.imread(image_path)

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Tạo thư mục output dựa trên tên của ảnh
        output_folder = f'output_images_{image_name}'
        os.makedirs(output_folder, exist_ok=True)

        # Cắt ảnh dựa trên bounding box và lưu vào thư mục output
        for i, bounding_box in enumerate(data_list):
            # Chuyển đổi danh sách tọa độ thành tuple và kiểu dữ liệu integer
            bounding_box = [(int(point[0]), int(point[1]))
                            for point in bounding_box]
            # Cắt ảnh theo bounding box
            roi = image[bounding_box[0][1]:bounding_box[2]
                        [1], bounding_box[0][0]:bounding_box[2][0]]
            # Đường dẫn để lưu ảnh
            umber_second = int(bounding_box[3][0])
            output_path = os.path.join(
                output_folder, f'output_{umber_second}.jpg')
            # Lưu ảnh vào thư mục output
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

        # Mở file đầu vào và đọc nội dung
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Mở file đầu ra để ghi dữ liệu đã sắp xếp
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in lines:
                data = eval(line)
                sorted_data = sorted(data, key=lambda x: int(
                    re.search(r'output_(\d+).jpg', x[0]).group(1)))
                for item in sorted_data:
                    output_file.write(item[1][0] + ' ')
                output_file.write('\n')

        # Tạo hình ảnh từ file văn bản
        image_size = (800, 600)  # Kích thước hình ảnh
        background_color = (255, 255, 255)  # Màu nền (trắng)
        text_color = (0, 0, 0)  # Màu văn bản (đen)
        font_size = 20  # Kích thước font
        font_path = "SVN-Arial-Regular.ttf"  # Đường dẫn đến file font

        image = Image.new('RGB', image_size, background_color)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)

        with open(output_file_path, 'r', encoding='utf-8') as output_file:
            lines = output_file.readlines()
            for i, line in enumerate(lines):
                draw.text((10, i * font_size), line.strip(),
                          text_color, font=font)

        # Lưu hình ảnh
        image.save(image_output_path)
        det_result_image = cv2.imread(
            f"inference_results\\det_res_{image_name}.jpg")
        # Mở hình ảnh bằng OpenCV
        image_cv2 = cv2.imread(image_output_path)
        cv2.imshow('Sorted Output Image', image_cv2)
        cv2.imshow("Detected Results", det_result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Hiển thị kết quả lên giao diện
        self.result_label.config(text="OCR hoàn thành.")


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR App")
        self.root.geometry("500x750")  # Đặt kích thước cửa sổ
        self.inference = Inference(self)
        self.image_path = None

        self.select_image_button = tk.Button(
            root, text="Chọn ảnh", command=self.load_image)
        # Thêm khoảng cách từ nút chọn ảnh đến cạnh trên
        self.select_image_button.pack(pady=20)

        self.start_ocr_button = tk.Button(
            root, text="Bắt đầu OCR", command=self.run_ocr)
        # Thêm khoảng cách từ nút bắt đầu OCR đến cạnh trên
        self.start_ocr_button.pack(pady=10)

        self.image_label = tk.Label(root)
        # Thêm khoảng cách từ ảnh đến các cạnh và set expand=True
        self.image_label.pack(padx=20, pady=20, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            image = Image.open(self.image_path)
            # Lấy kích thước của cửa sổ
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()
            # Thay đổi kích thước ảnh để phù hợp với kích thước cửa sổ
            image = image.resize(
                (window_width - 40, window_height - 100), Image.Resampling.LANCZOS)
            image = ImageTk.PhotoImage(image)
            self.image_label.config(image=image)
            self.image_label.image = image

    def run_ocr(self):
        if self.image_path:
            self.inference.infer(self.image_path)
        else:
            self.result_label.config(
                text="Hãy chọn ảnh trước khi bắt đầu OCR.")


if __name__ == '__main__':
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
