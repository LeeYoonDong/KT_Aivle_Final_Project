# 통합 민원처리 시스템

## 프로젝트 개요
이 프로젝트는 행정안전부의 DaaS(Desktop as a Service) 환경에서 운영되는 통합 민원처리 시스템입니다. 법령 정보를 수집, 저장, 분석하고 이를 바탕으로 지능형 민원 상담 서비스와 보안 관제 시스템을 제공합니다. 주요 기능으로는 법령 데이터베이스 구축, AI 기반 법률 분류, 챗봇 인터페이스, OCR을 이용한 문서 처리, 그리고 실시간 보안 관제 대시보드가 있습니다.

## 시스템 아키텍처
1. **데이터 계층**
   - 법령 데이터베이스 (MySQL)
   - 보안 로그 데이터베이스

2. **애플리케이션 계층**
   - 법령 데이터 수집 및 관리 모듈
   - AI 기반 법률 분류 엔진
   - 자연어 처리 챗봇 엔진
   - OCR 및 개인정보 마스킹 모듈
   - 실시간 보안 모니터링 엔진

3. **프레젠테이션 계층**
   - 통합 민원처리 인터페이스 (PyQt5 기반)
   - 보안 관제 대시보드 (Streamlit 기반)

## 주요 구성 요소

1. **법령 데이터베이스 (law_DB.py)**
   - 국가법령정보센터 API를 활용한 법령 데이터 수집
   - MySQL 데이터베이스에 구조화된 형태로 저장
   - 법령 기본 정보 및 조문 데이터 관리

2. **지능형 민원 상담 서비스 (LawChatBot.py)**
   - BERT 기반 법률 분류 모델 활용
   - OpenAI GPT를 이용한 자연어 응답 생성
   - 사용자 질의에 대한 관련 법령 정보 제공

3. **통합 민원처리 인터페이스 (main.py)**
   - 지능형 민원 상담 서비스와 문서 처리 기능 통합
   - 민원 서류 업로드 및 자동 분석 기능

4. **문서 처리 및 개인정보 보호 (OcrMsk.py)**
   - OCR을 이용한 민원 서류 텍스트 추출
   - 개인정보 자동 탐지 및 마스킹 처리

5. **실시간 보안 관제 대시보드 (SecurityControlDashboard.py)**
   - DaaS 환경의 실시간 보안 상태 모니터링
   - 사용자 활동, 시스템 리소스, 보안 이벤트 등 종합적 정보 제공
   - 일별/월별 접속자 추이, 보안 정책 준수율 등 시각화

6. **AI 모델 학습 (train_model.py)**
   - BERT 기반 법률 분류 모델 학습 및 최적화
   - 데이터베이스에서 법령 데이터 로드 및 전처리
   - 모델 성능 평가 및 개선

## 주요 기능 및 특징
1. **지능형 민원 상담**
   - AI 기반 법령 분석 및 관련 정보 제공
   - 자연어 처리를 통한 맞춤형 답변 생성

2. **민원 서류 자동 처리**
   - OCR을 통한 문서 디지털화
   - 개인정보 자동 탐지 및 마스킹

3. **실시간 보안 관제**
   - DaaS 환경의 종합적 보안 상태 모니터링
   - 이상 징후 실시간 탐지 및 알림

4. **데이터 기반 의사결정 지원**
   - 민원 처리 현황 및 트렌드 분석
   - 보안 정책 준수율 등 주요 지표 제공

## 시스템 요구사항
- DaaS(Desktop as a Service) 환경
- Python 3.x
- MySQL 서버
- PyTorch, Transformers (Hugging Face)
- OpenAI API
- PyQt5, Streamlit
- EasyOCR
- 기타 필요 라이브러리: pandas, numpy, sklearn, configparser 등

## 설치 및 설정
1. 필요한 Python 패키지 설치:
   ```
   pip install -r requirements.txt
   ```
2. MySQL 데이터베이스 설정
3. `config.ini` 파일에 데이터베이스 및 API 키 정보 입력
4. 법령 데이터 수집 및 데이터베이스 구축:
   ```
   python law_DB.py
   ```
5. 법률 분류 모델 학습:
   ```
   python train_model.py
   ```

## 사용 방법
1. 통합 민원처리 시스템 실행:
   ```
   python main.py
   ```
2. 보안 관제 대시보드 실행:
   ```
   streamlit run SecurityControlDashboard.py
   ```

## 보안 및 규정 준수
- 개인정보보호법 준수
- 행정안전부 보안 정책 적용
- 실시간 보안 모니터링 및 위협 대응

## 향후 계획
1. AI 모델 성능 지속적 개선
2. 타 행정 시스템과의 연계 확대
3. 사용자 피드백 기반 서비스 개선


<br/>
<br/>

# INTEGRATED CUSTOMER PROCESSING SYSTEM

## Project Overview
This project is an integrated civil complaint handling system that operates in the DaaS (Desktop as a Service) environment of the Ministry of Public Administration and Security. It collects, stores, analyzes, and provides intelligent civil complaint counseling services and security control systems based on the legal information. Its main functions include the establishment of a legal database, AI-based legal classification, chatbot interface, document processing using OCR, and real-time security control dashboard.

## System Architecture
1. **Data Hierarchy**
- Statutory Database (MySQL)
- Security Log Database

2. **Application layer**
- Statutory Data Collection and Management Module
- AI-based legal classification engine
- Natural language processing chatbot engine
- OCR and Personal Information Masking Module
- Real-time Security Monitoring Engine

3. **Hierarchy of presentations**
- Integrated Complaint Handling Interface (Based on PyQt5)
- Security Control Dashboard (Based on Streamlit)

## Key components

1. **Legislation database (law_DB.py)**
- National Legal Information Center API for Statutory Data Collection
- Save to MySQL database in a structured form
- Managing Statutory Basic Information and Articles Data

2. **Intelligent civil service (LawChatBot.py )**
- Utilize BERT-based legal classification model
- Generating natural language responses using OpenAI GPT
- Providing relevant statutory information for user queries

3. **Integrated Complaint Handling Interface (main.py )**
- Integrating intelligent civil service and document processing capabilities
- Upload and Auto-Analyze Complaint

4. **Document processing and privacy (OcrMsk.py )**
- Text extraction of complaints documents using OCR
- Automatic detection and masking of personal information

5. **Real-time Security Control Dashboard (SecurityControlDashboard.py )**
- Monitor real-time security health of DaaS environments
- Provides comprehensive information, including user activities, system resources, and security events
- Visualize daily/monthly user trends, security policy compliance, etc

6. **Learning AI models (train_model.py)**
- Learning and optimizing BERT-based legal classification models
- Loading and preprocessing statutory data from the database
- Evaluate and improve model performance

## Key Features and Features
1. **Intelligent Complaint Consultation**
- AI-based legal analysis and relevant information
- Generating customized answers through natural language processing

2. **Automatic handling of complaints documents**
- Digitize documents with OCR
- Automatic detection and masking of personal information

3. **Real-time security control**
- Monitor comprehensive security health of DaaS environments
- Real-time anomaly detection and notification

4. **Supports data-driven decision-making**
- Analysis of Civil Complaint Handling Status and Trends
- Provides key indicators such as security policy compliance

## System Requirements
- DaaS(Desktop as a Service) 환경
- Python 3.x
- MySQL server
- PyTorch, Transformers (Hugging Face)
- OpenAI API
- PyQt5, Streamlit
- EasyOCR
- Other required libraries: pandas, numpy, sklearn, configparser, etc

## Installation and Settings
1. Install the required Python package:
```
pip install -r requirements.txt
```
2. MySQL database settings
3. Enter database and API key information in the file 'config.ini'
4. Statutory data collection and database construction:
```
python law_DB.py
```
5. Learn the legal classification model:
```
python train_model.py
```

## How to use it
1. Integrated Complaint Handling System Runs:
```
python main.py
```
2. Run the Security Control Dashboard:
```
streamlit run SecurityControlDashboard.py
```

## Security and compliance
- Compliance with Personal Information Protection Act
- Applying the Security Policy of the Ministry of Public Administration and Security
- Real-time security monitoring and threat response

## Future plans
1. Continuous improvement in AI model performance
2. Expansion of linkage with other administrative systems
3. Improve user feedback-based service
