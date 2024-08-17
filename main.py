import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QTextCursor, QColor
from PyQt5.QtCore import Qt

from LawChatBot import ChatbotWindow, get_config, search_law, ask_gpt
from OcrMsk import FileProcessorApp

class IntegratedSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.chatbot = ChatbotWindow(self.config)
        self.file_processor = FileProcessorApp()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('통합 민원 처리 시스템')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.chatDisplay = QTextEdit()
        self.chatDisplay.setReadOnly(True)
        layout.addWidget(self.chatDisplay)

        self.dropLabel = QLabel('파일을 여기에 끌어다 놓으세요')
        self.dropLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.dropLabel)

        uploadButton = QPushButton('민원 서류 업로드')
        uploadButton.clicked.connect(self.uploadDocument)
        layout.addWidget(uploadButton)

        inputLayout = QHBoxLayout()
        self.inputBox = QTextEdit()
        self.inputBox.setFixedHeight(50)
        self.inputBox.installEventFilter(self)
        inputLayout.addWidget(self.inputBox)

        sendButton = QPushButton('전송')
        sendButton.clicked.connect(self.sendMessage)
        inputLayout.addWidget(sendButton)

        layout.addLayout(inputLayout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            self.process_file(url.toLocalFile())

    def uploadDocument(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "민원 서류 선택", "", "PDF Files (*.pdf);;Image Files (*.png *.jpg *.jpeg)")
        if fileName:
            self.process_file(fileName)

    def process_file(self, file_path):
        ocr_text = self.file_processor.process_file(file_path)
        if ocr_text:
            self.displayMessage("민원 서류 내용:", "System", color=QColor("purple"))
            self.displayMessage(ocr_text, "System", color=QColor("purple"))

            law_info = search_law(ocr_text, self.chatbot.connection, self.chatbot.model, self.chatbot.tokenizer)
            if law_info:
                response = ask_gpt(ocr_text, law_info, self.config['api']['openai_key'])
                self.displayMessage("관련 법안 정보:", "System", color=QColor("green"))
                self.displayMessage(response, "System", color=QColor("green"))
            else:
                self.displayMessage("관련 법안을 찾을 수 없습니다.", "System", color=QColor("red"))

    def sendMessage(self):
        query = self.inputBox.toPlainText().strip()
        if not query:
            return
        self.displayMessage(query, "User", align="right", color=QColor(" skyblue"))
        self.inputBox.clear()

        law_info = search_law(query, self.chatbot.connection, self.chatbot.model, self.chatbot.tokenizer)
        if law_info:
            response = ask_gpt(query, law_info, self.config['api']['openai_key'])
            self.displayMessage(response, "Assistant", align="left", color=QColor("yellow"))
        else:
            self.displayMessage("죄송합니다. 관련 법률 정보를 찾을 수 없습니다.", "Assistant", align="left", color=QColor("skyblue"))

    def displayMessage(self, message, sender, align="left", color=QColor("black")):
        cursor = self.chatDisplay.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chatDisplay.setTextCursor(cursor)

        format = self.chatDisplay.currentCharFormat()
        format.setForeground(color)

        if align == "right":
            self.chatDisplay.setAlignment(Qt.AlignRight)
        else:
            self.chatDisplay.setAlignment(Qt.AlignLeft)

        self.chatDisplay.setCurrentCharFormat(format)
        self.chatDisplay.insertPlainText(f"{sender}: {message}\n\n")
        self.chatDisplay.ensureCursorVisible()

    def eventFilter(self, source, event):
        if (source is self.inputBox and event.type() == event.KeyPress and
                event.key() == Qt.Key_Return and event.modifiers() != Qt.ShiftModifier):
            self.sendMessage()
            return True
        return super().eventFilter(source, event)

def main():
    app = QApplication(sys.argv)
    ex = IntegratedSystem()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
