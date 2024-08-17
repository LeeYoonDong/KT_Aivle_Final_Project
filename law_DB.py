import requests
import xml.etree.ElementTree as ET
import mysql.connector
from mysql.connector import Error
import configparser
import sys


def get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config


def get_law_data(law_id):
    url = "https://www.law.go.kr/DRF/lawService.do"
    params = {
        "OC": "ngho1202",
        "target": "law",
        "MST": law_id,
        "type": "XML"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        if "<?xml" not in response.text:
            raise ValueError("Response is not valid XML")
        return response.text
    except requests.RequestException as e:
        print(f"API 요청 중 오류 발생: {e}")
        return None


def check_database_exists(cursor, db_name):
    cursor.execute("SHOW DATABASES")
    databases = [db[0] for db in cursor]
    return db_name in databases


def create_database_if_not_exists(cursor, db_name):
    if check_database_exists(cursor, db_name):
        print(f"데이터베이스 '{db_name}'가 이미 존재합니다. 해당 데이터베이스를 사용합니다.")
        return True
    try:
        cursor.execute(f"CREATE DATABASE {db_name}")
        print(f"데이터베이스 '{db_name}'가 생성되었습니다.")
        return True
    except Error as e:
        print(f"데이터베이스 생성 중 오류 발생: {e}")
        return False


def create_tables(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS law_categories (
                id INT PRIMARY KEY AUTO_INCREMENT,
                name VARCHAR(100)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS laws (
                id VARCHAR(20) PRIMARY KEY,
                name TEXT,
                category_id INT,
                publish_date DATE,
                effective_date DATE,
                ministry TEXT,
                version INT,
                FOREIGN KEY (category_id) REFERENCES law_categories(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INT PRIMARY KEY AUTO_INCREMENT,
                law_id VARCHAR(20),
                article_number VARCHAR(20),
                content TEXT,
                effective_date DATE,
                embedding JSON,
                FOREIGN KEY (law_id) REFERENCES laws(id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INT PRIMARY KEY AUTO_INCREMENT,
                keyword VARCHAR(100)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS law_keywords (
                law_id VARCHAR(20),
                keyword_id INT,
                PRIMARY KEY (law_id, keyword_id),
                FOREIGN KEY (law_id) REFERENCES laws(id),
                FOREIGN KEY (keyword_id) REFERENCES keywords(id)
            )
        """)
        conn.commit()
        print("테이블이 생성되었거나 이미 존재합니다.")
    except Error as e:
        print(f"테이블 생성 중 오류 발생: {e}")


def process_law_data(xml_data, conn):
    if xml_data is None:
        print("XML 데이터가 없습니다.")
        return 0

    root = ET.fromstring(xml_data)
    cursor = conn.cursor()
    processed_count = 0

    try:
        basic_info = root.find('기본정보')
        law_id = basic_info.find('법령ID').text

        # laws 테이블에 데이터 삽입
        cursor.execute("""
            INSERT INTO laws (id, name, publish_date, effective_date, ministry, version)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            name = VALUES(name),
            publish_date = VALUES(publish_date),
            effective_date = VALUES(effective_date),
            ministry = VALUES(ministry),
            version = version + 1
        """, (
            law_id,
            basic_info.find('법령명_한글').text,
            basic_info.find('공포일자').text,
            basic_info.find('시행일자').text,
            basic_info.find('소관부처').text,
            1  # 초기 버전
        ))

        # articles 테이블에 데이터 삽입
        if law_id == '239293':
            for item in root.findall('.//본문/조문'):
                article_number = item.find('조문번호').text if item.find('조문번호') is not None else ''
                article_content = item.find('조문내용').text if item.find('조문내용') is not None else ''

                cursor.execute("""
                    INSERT INTO articles (law_id, article_number, content, effective_date)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    content = VALUES(content),
                    effective_date = VALUES(effective_date)
                """, (
                    law_id,
                    article_number,
                    article_content,
                    basic_info.find('시행일자').text
                ))
                processed_count += 1
        else:
            for article in root.findall('.//조문단위'):
                cursor.execute("""
                    INSERT INTO articles (law_id, article_number, content, effective_date)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    content = VALUES(content),
                    effective_date = VALUES(effective_date)
                """, (
                    law_id,
                    article.find('조문번호').text,
                    article.find('조문내용').text,
                    article.find('조문시행일자').text
                ))
                processed_count += 1

        conn.commit()
        print(f"법령 ID {law_id} 처리 완료")
        return processed_count

    except Error as e:
        print(f"법률 데이터 처리 중 오류 발생: {e}")
        conn.rollback()
        return 0

def read_law_ids(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def clear_existing_data(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM articles")
        cursor.execute("DELETE FROM laws")
        conn.commit()
        print("기존 데이터가 모두 삭제되었습니다.")
    except Error as e:
        print(f"데이터 삭제 중 오류 발생: {e}")
        conn.rollback()

def main():
    config = get_config()
    db_config = config['database']

    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )

        cursor = conn.cursor()


        if not create_database_if_not_exists(cursor, db_config['db_name']):
            print("프로그램을 종료합니다.")
            return

        conn.database = db_config['db_name']

        create_tables(conn)

        # 사용자 입력 받기
        choice = input("기존 데이터를 어떻게 처리할까요? (1: 유지 및 업데이트, 2: 모두 삭제 후 새로 입력): ")

        if choice == '2':
            clear_existing_data(conn)

        law_ids = read_law_ids('law_ids.txt')
        processed_data = {}

        for law_id in law_ids:
            try:
                xml_data = get_law_data(law_id)
                if xml_data:
                    count = process_law_data(xml_data, conn)
                    processed_data[law_id] = count
                    print(f"법령 ID {law_id} 처리 완료: {count}개의 데이터 처리됨")
                else:
                    print(f"법령 ID {law_id} 데이터를 가져오는데 실패했습니다.")
            except Exception as e:
                print(f"법령 ID {law_id} 처리 중 오류 발생: {str(e)}")

        print("\n각 법령 ID별 처리된 데이터 수:")
        for law_id, count in processed_data.items():
            print(f"법령 ID {law_id}: {count}개")

    except Error as e:
        print(f"DB 연결 중 오류 발생: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            print("MySQL 연결 종료")

if __name__ == "__main__":
    main()