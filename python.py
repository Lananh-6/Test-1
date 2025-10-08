# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

# 1. Định nghĩa CSS tùy chỉnh để áp dụng cho tiêu đề chính (h1)
custom_css = """
<style>
/* Áp dụng cho tiêu đề chính */
h1 {
    color: #1E90FF; /* Màu xanh dương (Dodger Blue) */
    text-transform: uppercase; /* In hoa */
    font-family: 'Times New Roman', Times, serif; /* Font chữ */
    font-size: 2.5em; /* Giữ kích thước tương đương st.title */
}
</style>
"""

# Chèn CSS vào ứng dụng
st.markdown(custom_css, unsafe_allow_html=True)

# 2. Thay thế st.title bằng st.markdown() để sử dụng tiêu đề tùy chỉnh
# st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊") # DÒNG CŨ ĐÃ BỊ THAY THẾ
st.markdown("<h1>Ứng dụng Phân Tích Báo Cáo Tài Chính 📊</h1>", unsafe_allow_html=True)

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất)
# --- (GIỮ NGUYÊN)
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""

    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]

    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU
    # *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC
    # *******************************

    return df

# --- Hàm gọi API Gemini (Dùng cho chức năng phân tích) ---
# --- (GIỮ NGUYÊN)
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận
    xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính
        sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình
        hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng,
        thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.

        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# ==============================================================================
#                 CHỨC NĂNG CHAT (GIỮ NGUYÊN)
# ==============================================================================

def initialize_chat():
    """Khởi tạo Client, Phiên Chat (có lịch sử) và session state."""
    # 1. Kiểm tra và lấy API Key
    if "api_key" not in st.session_state:
        st.session_state.api_key = st.secrets.get("GEMINI_API_KEY")

    if not st.session_state.api_key:
        return None, "API Key not found"

    try:
        # 2. Khởi tạo Client
        if "client" not in st.session_state:
            st.session_state.client = genai.Client(api_key=st.session_state.api_key)
            st.session_state.model_name = 'gemini-2.5-flash'

        # 3. Khởi tạo Chat Session (duy trì lịch sử hội thoại)
        if "chat_session" not in st.session_state:
            system_instruction = (
                "Bạn là một trợ lý AI chuyên nghiệp về Phân tích Tài chính và Kế toán. "
                "Hãy giải đáp các thắc mắc về các khái niệm, chỉ số, phương pháp phân tích, "
                "và kiến thức tài chính nói chung. Trả lời bằng Tiếng Việt."
            )
            
            st.session_state.chat_session = st.session_state.client.chats.create(
                model=st.session_state.model_name,
                config
