
import mysql.connector
from mysql.connector import Error
import configparser
from openai import OpenAI
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QTextCursor, QColor
from PyQt5.QtCore import Qt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification


def get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config


def connect_to_database(db_config):
    try:
        connection = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['db_name']
        )
        return connection
    except Error as e:
        print(f"데이터베이스 연결 오류: {e}")
        return None


def load_model(model_path, num_labels):
    model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict(text, model, tokenizer, max_len=256):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

    return preds.item()


def search_law(query, connection, model, tokenizer):
    try:
        cursor = connection.cursor()
        cursor.execute("""
        SELECT l.id, l.name, a.article_number, a.content
        FROM laws l
        JOIN articles a ON l.id = a.law_id
        """)
        results = cursor.fetchall()

        predicted_label = predict(query, model, tokenizer)

        relevant_laws = [law for law in results if law[0] == predicted_label]

        if not relevant_laws:
            # 예측된 법률이 없으면 기존의 TF-IDF 방식으로 검색
            documents = [f"{row[1]} {row[2]} {row[3]}" for row in results]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents + [query])
            cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            top_3_indices = cosine_similarities.argsort()[-3:][::-1]
            relevant_laws = [results[idx] for idx in top_3_indices]

        return [(law[0], law[1], law[2], law[3][:200] + "...", 1.0) for law in relevant_laws[:3]]

    except Error as e:
        print(f"검색 오류: {e}")
        return []


def ask_gpt(query, law_info, api_key):
    client = OpenAI(api_key=api_key)
    prompt = f"사용자 질문: {query}\n\n관련 법률 정보:\n"
    for law in law_info:
        prompt += f"법령ID: {law[0]}, 법령명: {law[1]}, 조항: {law[2]}\n내용: {law[3]}\n유사도: {law[4]:.2f}\n\n"
    prompt += "위 정보를 바탕으로 사용자의 질문에 답변해주세요. 관련 법령의 이름과 내용을 인용하여 설명해주세요."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that provides information about Korean laws."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT 응답 생성 중 오류 발생: {str(e)}"


class ChatbotWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.connection = connect_to_database(config['database'])

        # 모델 및 토크나이저 로드
        self.tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")
        cursor = self.connection.cursor()
        cursor.execute("SELECT DISTINCT id FROM laws")
        num_labels = len(cursor.fetchall())
        self.model = load_model('bert_law_classifier.pth', num_labels)

        self.initUI()

    def initUI(self):
        self.setWindowTitle('법률 정보 챗봇')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.chatDisplay = QTextEdit()
        self.chatDisplay.setReadOnly(True)
        layout.addWidget(self.chatDisplay)

        inputLayout = QHBoxLayout()
        self.inputBox = QTextEdit()
        self.inputBox.setFixedHeight(50)
        self.inputBox.installEventFilter(self)
        inputLayout.addWidget(self.inputBox)

        layout.addLayout(inputLayout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def eventFilter(self, source, event):
        if (source is self.inputBox and event.type() == event.KeyPress and
                event.key() == Qt.Key_Return and event.modifiers() != Qt.ShiftModifier):
            self.sendMessage()
            return True
        return super().eventFilter(source, event)

    def sendMessage(self):
        query = self.inputBox.toPlainText().strip()
        if not query:
            return
        self.displayMessage(query, "Human", align="right", color=QColor("blue"))
        self.inputBox.clear()

        law_info = search_law(query, self.connection, self.model, self.tokenizer)
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

    def closeEvent(self, event):
        if self.connection:
            self.connection.close()
        event.accept()