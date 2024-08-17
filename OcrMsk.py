"""
OCR동작 구조
A(실행)
    1.인터페이스 출력
    2.실행시 드래그&드랍 or 파일 서치로  목표 파일 등록
    2-1. 사용가능 파일은 PDF, IMG( .png, .jpg, .jpeg)
B(마스킹)
    3.파일 업로드시 확장자로 파일 구분해 확장자에 맞는 함수 동작
    3-1.PDF: 모든 페이지 이미지화 -> ocr실행
    3-2.IMG: ocr 실행
    4.개인정보 패턴과 일치 하는 대상 식별
    5.식별된 영역 바운딩 박스 생성
    6.박스 영역 흑색 마스킹
    7.마스킹된 이미지 리턴
C(저장)
    8.저장 디렉토리의 파일명을 현재 '월/일' 로 하여 폴더 생성(중복생성 불가)
    9.8에서 생성된 폴더 내부 '원본파일명'+'msk'+'HH%MM%SS'의 형식으로 저장
"""
import os
from datetime import datetime
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtCore import Qt
import easyocr
import cv2
import numpy as np
import re
from pdf2image import convert_from_path
import torch

class FileProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.reader = easyocr.Reader(['ko', 'en'], gpu=torch.backends.mps.is_available())
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle('개인정보 마스킹 도구')
        self.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout()
        self.drop_label = QLabel('파일을 여기에 끌어다 놓으세요')
        self.drop_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.drop_label)

        self.select_button = QPushButton('파일 선택')
        self.select_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.select_button)

        self.setLayout(layout)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            self.process_file(url.toLocalFile())

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '파일 선택', '', 'Images (*.png *.jpg *.jpeg);;PDF Files (*.pdf)')
        if file_path:
            return self.process_file(file_path)

    def process_file(self, file_path):
        output_dir = self.create_output_directory()
        output_path = self.generate_output_path(file_path, output_dir)

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension in ['.png', '.jpg', '.jpeg']:
            return self.process_image(file_path, output_path)
        elif file_extension == '.pdf':
            return self.process_pdf(file_path, output_path)
        else:
            print(f"지원되지 않는 파일 형식: {file_extension}")
            return None

    @staticmethod
    def create_output_directory():
        today = datetime.now().strftime("%m%d")
        output_dir = os.path.join('ocr test/output', today)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def generate_output_path(input_path, output_dir):
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        timestamp = datetime.now().strftime("%H%M%S")
        return os.path.join(output_dir, f"{name}_msk_{timestamp}{ext}")

    def process_image(self, input_path, output_path):
        image = cv2.imread(input_path)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {input_path}")
            return None

        results = self.reader.readtext(image)
        masked_image = self.mask_personal_info(image, results)
        cv2.imwrite(output_path, masked_image)
        print(f"이미지 처리 완료: {output_path}")
        return ' '.join([text for _, text, _ in results])

    def process_pdf(self, input_path, output_path):
        images = convert_from_path(input_path)
        all_text = []
        for i, image in enumerate(images):
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            results = self.reader.readtext(image_bgr)
            masked_image = self.mask_personal_info(image_bgr, results)
            page_output = f"{output_path.rsplit('.', 1)[0]}_page_{i+1}.png"
            cv2.imwrite(page_output, masked_image)
            print(f"PDF 페이지 {i+1} 처리 완료: {page_output}")
            all_text.extend([text for _, text, _ in results])
        return ' '.join(all_text)

    def mask_personal_info(self, image, results):
        patterns = {
            'kor_name': r'^(김|이|박|최|정|노)[가-힣]{2}$',
            'eng_name':r'\b[A-Z][a-z]+(?:[ \'-.][A-Z][a-z]+)*\b',
            'reg_num': r'\d{6}-\d{7}',
            'birth_date': r'\d{6}',
            'phone': r'(\+?\d{1,2}[-\s]?)?(\(?\d{3}\)?[-\s]?)?\d{3}[-\s]?\d{4}|010-\d{4}-\d{4}'
        }
        regions = ['경기', '충청', '경상', '전라', '제주', '세종', '서울', '인천', '대전', '대구', '광주', '부산', '울산']

        for i, (bbox, text, _) in enumerate(results):
            if any(re.search(pattern, text) for pattern in patterns.values()):
                self.draw_mask(image, bbox)

            if any(region in text for region in regions):
                self.mask_address(image, results, i, bbox)

        return image

    @staticmethod
    def draw_mask(image, bbox):
        cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 0, 0), -1)

    def mask_address(self, image, results, start_index, start_bbox):
        for j in range(start_index, len(results)):
            curr_bbox = results[j][0]
            if curr_bbox[0][1] == start_bbox[0][1] and curr_bbox[0][0] >= start_bbox[0][0]:
                if j == start_index or curr_bbox[0][0] - start_bbox[2][0] <= 50:
                    self.draw_mask(image, curr_bbox)
                else:
                    break