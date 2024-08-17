import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from PIL import Image
import random
import streamlit.components.v1 as components

st.set_page_config(page_title="행정안전부 DaaS 보안 관제 시스템", layout="wide")

# CSS
st.markdown("""
<style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 250px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 250px;
        margin-left: -250px;
    }
    .left-aligned-title {
    text-align: left;
    color: white;
    margin-left: -130px;  # 필요에 따라 조정
    }
    .reportview-container .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    .stPlotlyChart {
        height: 200px;
    }
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2{
        font-size: 1.2rem;
    }
    .st-emotion-cache-16txtl3 p, .st-emotion-cache-16txtl3 div {
        font-size: 0.8rem;
    }
    .dataframe {
        font-size: 0.7rem;
    }
    .st-bx {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0;
        background-color: #1e2130;
        color: white;
        border: none;
    }
    .content-box {
        border: 1px solid #4c5866;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #1e2130;
    ##
    .content-box p, .content-box div, .content-box span {
        font-size: 1.6rem !important;
    }
    .stTable td, .stTable th {
        font-size: 1.4rem !important;
    }
    .st-bb, .st-af, .st-ag, .st-ae, .st-bq {
        font-size: 1.6rem !important;
    }
    .stPlotlyChart {
        height: 400px !important;
    }    

    }

</style>
""", unsafe_allow_html=True)

# 사이드바 (좌)
with st.sidebar:
    st.title("통합 보안 관제 목록")
    st.button("통합 모니터링", key="dashboard")
    departments = [
        "사이버 보안관제",
        "개인정보 보호",
        "침해사고 대응",
        "취약점 분석",
        "보안정책 관리",
        "접근 제어",
        "암호화 관리",
        "보안 교육",
        "재해 복구",
        "네트워크 보안",
        "엔드포인트 보안",
        "클라우드 보안",
        "데이터 유출 방지",
        "컴플라이언스"
    ]
    # 로그 분석 버튼 추가
    if st.button("로그 분석", key="log_analysis"):
        st.session_state.show_log_analysis = True
    else:
        st.session_state.show_log_analysis = False

    # 보안 관제 버튼 추가
    if st.button("보안분석 위험현황", key="cyber_security"):
        st.session_state.show_cyber_security = True
    else:
        st.session_state.show_cyber_security = False

        # 기존의 다른 버튼들
    for i, dept in enumerate(departments):
        if dept == "개인정보 보호":
            if st.button(dept, key=f"button_{i}"):
                st.session_state.show_pdf = not st.session_state.show_pdf
        else:
            st.button(dept, key=f"button_{i}")

# 타이틀 이미지 추가
col_img, col_title = st.columns([1, 5])

with col_img:
    image = Image.open('img/NIS.png')
    st.image(image, width=120)

with col_title:
    st.markdown("<h1 class='left-aligned-title'>행정안전부 통합 보안 관제 모니터</h1>", unsafe_allow_html=True)

clock_html = """
<div id='clock' style='font-family: "IBM Plex Sans", sans-serif; font-size: 1.6rem; font-weight: 400; color: rgb(250, 250, 250);'></div>
<script>
function updateClock() {
    var now = new Date();
    var timeString = now.getFullYear() + "년 " + 
                     String(now.getMonth() + 1).padStart(2, '0') + "월 " + 
                     String(now.getDate()).padStart(2, '0') + "일 " + 
                     String(now.getHours()).padStart(2, '0') + "시 " + 
                     String(now.getMinutes()).padStart(2, '0') + "분 " + 
                     String(now.getSeconds()).padStart(2, '0') + '초';
    document.getElementById('clock').textContent = "현재 시간: " + timeString;
}
updateClock();
setInterval(updateClock, 1000);
</script>
"""

components.html(clock_html, height=50)

# 메인 컨텐츠


if 'show_log_analysis' in st.session_state and st.session_state.show_log_analysis:
    st.title("보안 로그 분석 대시보드")

    # 현재 날짜와 시간 가져오기
    current_time = datetime.now()

    # 상단 지표 (변경 없음)
    st.markdown("<h2 style='text-align: center;'>주요 지표</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Logins", "2064", "5")
    col2.metric("Sign ups", "263", "12")
    col3.metric("Sign outs", "268", "-8")
    col4.metric("Support calls", "80", "-3")


    # Memory/CPU 그래프
    def generate_memory_cpu_data(start_time):
        time = [start_time - timedelta(minutes=5 * i) for i in range(199, -1, -1)]
        memory = [random.uniform(0, 60) for _ in range(200)]
        cpu = [random.uniform(0, 25) for _ in range(200)]
        return pd.DataFrame({"time": time, "memory": memory, "cpu": cpu})


    memory_cpu_data = generate_memory_cpu_data(current_time)

    col1, col2 = st.columns(2)

    with col1:
        fig_memory = px.line(memory_cpu_data, x="time", y="memory",
                             labels={"memory": "Usage (%)", "time": "Time"},
                             title="Memory Usage")
        fig_memory.update_layout(
            height=400,
            legend_title_text='',
            margin=dict(l=50, r=50, t=50, b=50),
            yaxis_title="Usage (%)",
            xaxis_title="Time"
        )
        fig_memory.update_traces(mode="lines+markers", line_color="#1f77b4")  # 파란색 계열
        st.plotly_chart(fig_memory, use_container_width=True)

    with col2:
        fig_cpu = px.line(memory_cpu_data, x="time", y="cpu",
                          labels={"cpu": "Usage (%)", "time": "Time"},
                          title="CPU Usage")
        fig_cpu.update_layout(
            height=400,
            legend_title_text='',
            margin=dict(l=50, r=50, t=50, b=50),
            yaxis_title="Usage (%)",
            xaxis_title="Time"
        )
        fig_cpu.update_traces(mode="lines+markers", line_color="#ff7f0e")  # 주황색 계열
        st.plotly_chart(fig_cpu, use_container_width=True)

    # Logins 그래프와 Server Requests 그래프를 나란히 배치
    col1, col2 = st.columns(2)

    with col1:
        def generate_login_data(start_time):
            time = [start_time - timedelta(minutes=5 * i) for i in range(199, -1, -1)]
            logins = [random.randint(20, 80) for _ in range(200)]
            logins_1h = [random.randint(20, 80) for _ in range(200)]
            return pd.DataFrame({"time": time, "logins": logins, "logins_1h": logins_1h})


        login_data = generate_login_data(current_time)
        fig_logins = px.line(login_data, x="time", y=["logins", "logins_1h"],
                             labels={"value": "Count", "variable": "Type"},
                             title="Logins")
        fig_logins.update_layout(height=400, legend_title_text='',
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_logins, use_container_width=True)

    with col2:
        def generate_server_requests(start_time):
            time = [start_time - timedelta(minutes=5 * i) for i in range(199, -1, -1)]
            servers = ["web_server_01", "web_server_02", "web_server_03", "web_server_04"]
            data = {server: [random.randint(0, 150) for _ in range(200)] for server in servers}
            data["time"] = time
            return pd.DataFrame(data)


        server_data = generate_server_requests(current_time)
        fig_servers = px.area(server_data, x="time",
                              y=["web_server_01", "web_server_02", "web_server_03", "web_server_04"],
                              labels={"value": "Requests", "variable": "Server"},
                              title="Server Requests")
        fig_servers.update_layout(height=400, legend_title_text='',
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_servers, use_container_width=True)


    # Client Side Full Page Load 그래프
    def generate_page_load_data(start_time):
        time = [start_time - timedelta(minutes=20 * i) for i in range(49, -1, -1)]
        load_times = [random.uniform(0, 6) for _ in range(50)]
        return pd.DataFrame({"time": time, "load_time": load_times})


    load_data = generate_page_load_data(current_time)
    fig_load = px.bar(load_data, x="time", y="load_time",
                      labels={"load_time": "Load Time (s)"},
                      title="Client Side Full Page Load",
                      color="load_time",  # 색상을 load_time 값에 따라 변경
                      color_continuous_scale=px.colors.sequential.Viridis)  # Viridis 색상 팔레트 사용

    fig_load.update_layout(height=400, coloraxis_showscale=False)  # 색상 스케일 범례 숨기기
    st.plotly_chart(fig_load, use_container_width=True)

else:
    if st.session_state.show_cyber_security:
        st.title("보안분석 상세 위험현황")

        # 상단 지표
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=1620000,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "금일 탐지/차단 합계", 'font': {'size': 14}},
                gauge={'axis': {'range': [None, 2000000]},
                       'bar': {'color': "lightgreen"},
                       'steps': [
                           {'range': [0, 1000000], 'color': "lightgray"},
                           {'range': [1000000, 2000000], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1620000}}))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=18,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "탐지 취약점", 'font': {'size': 14}},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "red"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 100], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 18}}))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        from datetime import datetime, timedelta

        # 현재 날짜 계산
        today = datetime.now().date()



        with col3:
            st.markdown("<h3 style='margin-bottom: 0px;'>금일/전일 IPS 탐지 추이</h3>", unsafe_allow_html=True)

            # 15일치 데이터 생성 (오늘 포함 15일)
            dates = [(today - timedelta(days=14 - i)) for i in range(15)]
            values_detected = [random.randint(200, 500) for _ in range(15)]
            values_blocked = [random.randint(100, 300) for _ in range(15)]

            df = pd.DataFrame({
                'Date': dates,
                '탐지': values_detected,
                '차단': values_blocked
            })

            fig = px.line(df, x='Date', y=['탐지', '차단'], markers=True)
            fig.update_layout(
                height=250,  # 그래프 높이 줄임
                margin=dict(l=20, r=20, t=10, b=20),  # 상단 여백 줄임
                xaxis_title="",
                yaxis_title="",
                legend_title_text="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(
                tickmode='array',
                tickvals=dates,
                ticktext=[d.strftime("%m-%d") for d in dates],
                tickangle=45
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # 중간 부분
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h3 style='margin-bottom: 0px;'>최근 7일 공격자 서버 집중 위협(IP/기간)</h3>", unsafe_allow_html=True)
            data = {
                'IP': ['185.224.128.184', '45.134.144.140', '194.31.98.152', '91.241.19.84'],
                'Duration': ['7일', '5일', '3일', '2일'],
                'Attacks': [1000, 800, 600, 400],
                'Technique': ['SQL Injection', 'DDoS', 'Credential Stuffing', 'XSS'],
                'Details': [
                    'Destination IP: 10.0.0.1<br>Protocol: TCP<br>Source Port: 12345<br>Destination Port: 80<br>Payload: "GET /admin HTTP/1.1\\r\\nHost: example.com\\r\\n\\r\\n"',
                    'Destination IP: 8.8.8.8<br>Protocol: UDP<br>Source Port: 1234<br>Destination Port: 53<br>Payload: "\\x01\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x03www\\x06google\\x03com\\x00\\x00\\x01\\x00\\x01"',
                    'Destination IP: 192.168.0.1<br>Protocol: ICMP<br>Type: 8 (Echo Request)<br>Code: 0<br>Payload: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
                    'Destination IP: 198.51.100.2<br>Protocol: TCP<br>Source Port: 22<br>Destination Port: 1234<br>Payload: "\\x00\\x00\\x00\\x0C\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x01"'
                ]
            }
            df = pd.DataFrame(data)
            fig = px.treemap(df, path=['IP', 'Technique', 'Duration', 'Details'], values='Attacks',
                             color='Attacks', color_continuous_scale='RdYlGn_r',
                             hover_data=['IP', 'Technique', 'Duration', 'Attacks', 'Details'])
            fig.update_traces(textinfo="label+value")
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("<h3 style='margin-bottom: 0px;'>최근 7일 집중 공격자 주요국</h3>", unsafe_allow_html=True)
            countries = ['북한', '중국', '러시아', '기타']
            attacks = [1134000, 164200, 83000, 43402]
            fig = px.bar(x=countries, y=attacks)
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="",
                yaxis_title="",
                # xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

        # 여기에 여백 추가
        st.markdown("<br><br><br>", unsafe_allow_html=True)

        # 하단 테이블 (기존 코드 유지)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("7일 중 3일 업데이트 목록")
            data = {
                '날짜': ['2023-07-01', '2023-07-02', '2023-07-03'],
                '건수': [123402, 324152, 110423],
                '상세': ['View', 'View', 'View']
            }
            st.table(data)

        with col2:
            st.subheader("최근 공격자 IP 업데이트(최근 7일)")
            data = {
                'IP': ['192.168.1.100', '192.168.1.101', '192.168.1.102'],
                '위험도': ['High', 'Medium', 'Low'],
                '국가': ['DPRK', 'CN', 'RU']
            }
            st.table(data)
    else:
        # 보안 경고 섹션
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        col_alert1, col_alert2, col_alert3 = st.columns(3)
        with col_alert1:
            st.error("고위험 경고: 15건")
        with col_alert2:
            st.warning("중위험 경고: 47건")
        with col_alert3:
            st.info("저위험 경고: 123건")
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("VM 현황")
            st.markdown("<p style='font-size: 1.6rem;'>총: 497,000대</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.6rem;'>정상: 496,540대</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.6rem;'>미등록: 400대</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.6rem;'>의심: 48대</p>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.6rem;'>위험: 12대</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("보안 점수")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=92,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "Darkgray", 'thickness': 0.4},  # 두께를 0.6, 투명도 추가
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(255, 0, 0, 0.6)"},  # 빨간색 채도 낮춤
                        {'range': [50, 80], 'color': "rgba(255, 255, 0, 0.6)"},  # 노란색 채도 낮춤
                        {'range': [80, 100], 'color': "rgba(0, 255, 0, 0.6)"}],  # 초록색 채도 낮춤
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 92}

                }
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(size=16)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("시스템 현황")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("활성 세션", "337,448", delta=None, delta_color="normal")
                st.metric("CPU 사용률", "68%", delta=None, delta_color="normal")
            with col_res2:
                st.metric("메모리 사용률", "72%", delta=None, delta_color="normal")
                st.metric("스토리지 사용률", "65%", delta=None, delta_color="normal")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("최근 보안 이벤트")
            security_events = [
                {"시간": "5분 전", "이벤트": "다중 로그인 실패 탐지"},
                {"시간": "15분 전", "이벤트": "비정상적 파일 접근 시도"},
                {"시간": "30분 전", "이벤트": "의심스러운 관리자 활동"},
                {"시간": "1시간 전", "이벤트": "대량 데이터 전송 감지"},
            ]
            st.table(pd.DataFrame(security_events).set_index('시간'))
            st.markdown('</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        current_date = datetime.now()

        with col1:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)

            # 일일 접속자 추이
            st.subheader("일일 접속자 추이")
            hours = [f"{i:02d}:00" for i in range(24)]

            max_value = 496573
            min_value = 378200

            # 코어 시간대 (09시-18시) 값 생성
            core_hours_values = [random.randint(min_value, max_value) for _ in range(10)]

            # 08시와 19시 값 설정
            value_08 = int(max_value / 8)
            value_19 = int(max_value / 6)

            # 그 외 시간대 값 설정
            other_hours_value = random.randint(2000, 3000)

            # 전체 24시간 데이터 생성
            daily_values = [other_hours_value] * 24
            daily_values[8] = value_08
            daily_values[9:19] = core_hours_values
            daily_values[19] = value_19

            daily_data = pd.DataFrame({"시간": hours, "접속자 수": daily_values})

            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(x=daily_data['시간'], y=daily_data['접속자 수'], mode='lines+markers'))
            fig_daily.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="시간",
                yaxis_title="접속자 수",
                yaxis_range=[0, max_value * 1.1]
            )
            st.plotly_chart(fig_daily, use_container_width=True)

            # 월간 접속자 추이
            st.subheader("월간 접속자 추이")
            months = ["1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]

            # 평균 일일 접속자 수 계산
            avg_daily_users = sum(core_hours_values) / len(core_hours_values)

            # 월별 근무일수 (주말 제외, 공휴일 고려)
            working_days = [21, 20, 23, 22, 21, 22, 23, 22, 21, 22, 22, 21]
            holidays = [0, 4, 0, 0, 6, 0, 0, 7, 0, 0, 0, 6]

            monthly_values = []
            for month, work_days, holiday in zip(range(1, 13), working_days, holidays):
                if month <= current_date.month:
                    monthly_value = int(avg_daily_users * (work_days - holiday))
                    monthly_values.append(monthly_value)
                else:
                    monthly_values.append(0)

            monthly_data = pd.DataFrame({"월": months, "접속자 수": monthly_values})

            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Scatter(x=monthly_data['월'], y=monthly_data['접속자 수'], mode='lines+markers'))
            fig_monthly.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="월",
                yaxis_title="접속자 수",
                yaxis_range=[0, max(monthly_values) * 1.1]
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("사용자 활동")
            user_activity = {
                "유형": ["정상 사용자", "의심 사용자", "차단된 사용자", "관리자"],
                "수": [5189, 42, 3, 45]
            }
            df_user_activity = pd.DataFrame(user_activity)
            st.dataframe(df_user_activity, hide_index=True, use_container_width=True)

            st.subheader("개인정보 접근 로그")
            pii_access = {
                "유형": ["정상 접근", "비정상 접근 시도", "차단된 접근"],
                "수": [1234, 17, 5]
            }
            df_pii_access = pd.DataFrame(pii_access)
            st.dataframe(df_pii_access, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="content-box">', unsafe_allow_html=True)
            st.subheader("보안 정책 준수율")
            compliance_rate = {
                "정책": ["패스워드 정책", "접근 제어", "데이터 암호화", "로그 관리", "보안 교육 이수"],
                "준수율": [98, 99, 100, 97, 95]
            }
            df_compliance = pd.DataFrame(compliance_rate)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_compliance['정책'],
                y=[100] * len(df_compliance),
                name='미준수율',
                marker_color='rgba(255, 0, 0, 0.6)'  # 연한 분홍색
            ))
            fig.add_trace(go.Bar(
                x=df_compliance['정책'],
                y=df_compliance['준수율'],
                name='현재 준수율',
                marker_color='rgba(230, 255, 255)'  # 연한 민트색
            ))

            fig.update_layout(
                barmode='overlay',
                yaxis_title="준수율 (%)",
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font=dict(size=16)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # 실시간 로그
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.subheader("실시간 보안 로그")
        log_placeholder = st.empty()

        current_time = datetime.now()

        a = current_time - timedelta(minutes=20, seconds=20)
        b = current_time - timedelta(minutes=11, seconds=37)
        c = current_time - timedelta(minutes=3, seconds=50)
        d = current_time - timedelta(seconds=22)

        a = a.strftime("%Y-%m-%d %H:%M:%S >> 방화벽: 의심스러운 해외 IP에서의 접근 시도 차단")
        b = b.strftime("%Y-%m-%d %H:%M:%S >> IDS: 가능한 SQL 인젝션 공격 탐지 및 차단")
        c = c.strftime("%Y-%m-%d %H:%M:%S >> 인증 시스템: 특정 계정 다중 로그인 실패, 일시적 잠금 조치")
        d = d.strftime("%Y-%m-%d %H:%M:%S >> DLP: 대용량 개인정보 파일 외부 전송 시도 차단")

        log_data = [a, b, c, d]

        log_placeholder.code('\n'.join(log_data))
        st.markdown('</div>', unsafe_allow_html=True)