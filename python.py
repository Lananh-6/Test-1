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

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
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

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini (Dùng cho Chức năng 5: Nhận xét 1 lần) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
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

# ****************************** PHẦN BỔ SUNG KHUNG CHAT BẮT ĐẦU ******************************

# --- Thiết lập Chat client (Sử dụng Caching) ---
@st.cache_resource
def get_chat_client(api_key):
    """Khởi tạo và trả về đối tượng ChatClient của Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        # Sử dụng chat.new_chat() để tạo một phiên chat với lịch sử
        chat = client.chats.create(model='gemini-2.5-flash')
        return chat
    except Exception as e:
        st.error(f"Lỗi khởi tạo Gemini Client/Chat: {e}")
        return None

# --- Khởi tạo State (Lịch sử chat) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# ****************************** PHẦN BỔ SUNG KHUNG CHAT KẾT THÚC ******************************

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Chia layout chính
if uploaded_file is not None:
    # --- Code xử lý file được đưa vào khối try-except để đảm bảo an toàn ---
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # Tính chỉ số Thanh toán Hiện hành cho Chức năng 4 và 5
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
            except IndexError:
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except ZeroDivisionError:
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
                st.error("Lỗi chia cho 0 khi tính Chỉ số Thanh toán Hiện hành. Vui lòng kiểm tra dữ liệu Nợ Ngắn Hạn.")


            # Chuẩn bị Dữ liệu Phân tích (Sử dụng cho cả Nhận xét 1 lần và Chat)
            data_for_ai_markdown = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if thanh_toan_hien_hanh_N != "N/A" else "N/A", 
                    f"{thanh_toan_hien_hanh_N_1}" if thanh_toan_hien_hanh_N_1 != "N/A" else "N/A", 
                    f"{thanh_toan_hien_hanh_N}" if thanh_toan_hien_hanh_N != "N/A" else "N/A"
                ]
            }).to_markdown(index=False) 
            
            # Tách nội dung chính và khung chat
            col_main, col_chat = st.columns([2, 1])

            with col_main:
                # --- Chức năng 2 & 3: Hiển thị Kết quả ---
                st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
                st.dataframe(df_processed.style.format({
                    'Năm trước': '{:,.0f}',
                    'Năm sau': '{:,.0f}',
                    'Tốc độ tăng trưởng (%)': '{:.2f}%',
                    'Tỷ trọng Năm trước (%)': '{:.2f}%',
                    'Tỷ trọng Năm sau (%)': '{:.2f}%'
                }), use_container_width=True)
                 
                # --- Chức năng 4: Tính Chỉ số Tài chính ---
                st.subheader("4. Các Chỉ số Tài chính Cơ bản")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1} lần"
                    )
                with col2:
                    delta_value = None
                    if thanh_toan_hien_hanh_N != "N/A" and thanh_toan_hien_hanh_N_1 != "N/A":
                        delta_value = f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N} lần",
                        delta=delta_value
                    )
                
                # --- Chức năng 5: Nhận xét AI (Button) ---
                st.subheader("5. Nhận xét Tình hình Tài chính (AI) - Nhận xét nhanh")
                if st.button("Yêu cầu AI Phân tích"):
                    api_key = st.secrets.get("GEMINI_API_KEY") 
                     
                    if api_key:
                        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                            ai_result = get_ai_analysis(data_for_ai_markdown, api_key)
                            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                            st.info(ai_result)
                    else:
                        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

            # ****************************** KHUNG CHAT BẮT ĐẦU ******************************
            with col_chat:
                st.subheader("6. Chat với Gemini về Báo cáo 💬")
                
                api_key = st.secrets.get("GEMINI_API_KEY")
                if not api_key:
                    st.error("Vui lòng cấu hình Khóa 'GEMINI_API_KEY' để sử dụng chức năng chat.")
                else:
                    # Khởi tạo hoặc lấy Chat client đã được cache
                    chat = get_chat_client(api_key)

                    if chat:
                        # Thêm bối cảnh ban đầu cho cuộc trò chuyện (chỉ 1 lần)
                        if not st.session_state.messages:
                            # Prompt hệ thống ban đầu (để mô hình biết về dữ liệu)
                            system_prompt = f"""
                            Bắt đầu một phiên hỏi đáp. Bạn là một chuyên gia phân tích tài chính.
                            Bạn có trách nhiệm trả lời các câu hỏi về báo cáo tài chính đã được tải lên.
                            Dữ liệu tài chính mà người dùng đang làm việc là:
                            {data_for_ai_markdown}
                            Hãy giới thiệu ngắn gọn và yêu cầu người dùng đặt câu hỏi.
                            """
                            # Gửi prompt ban đầu dưới dạng tin nhắn "giả" để khởi tạo chat
                            try:
                                with st.spinner('Đang khởi tạo phiên chat...'):
                                    response = chat.send_message(system_prompt)
                                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                            except Exception as e:
                                st.error(f"Lỗi khởi tạo chat: {e}")
                                

                        # Hiển thị lịch sử tin nhắn
                        for message in st.session_state.messages:
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])

                        # Xử lý input của người dùng
                        if prompt := st.chat_input("Hỏi Gemini về báo cáo tài chính này..."):
                            st.session_state.messages.append({"role": "user", "content": prompt})
                            with st.chat_message("user"):
                                st.markdown(prompt)

                            # Gửi tin nhắn và nhận phản hồi
                            with st.chat_message("assistant"):
                                with st.spinner("Đang nghĩ..."):
                                    try:
                                        response = chat.send_message(prompt)
                                        st.markdown(response.text)
                                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                                    except APIError as e:
                                        st.error(f"Lỗi gọi Gemini API trong chat: {e}")
                                        st.session_state.messages.append({"role": "assistant", "content": f"Lỗi gọi API: {e}"})
                                    except Exception as e:
                                        st.error(f"Lỗi không xác định: {e}")
                                        st.session_state.messages.append({"role": "assistant", "content": f"Đã xảy ra lỗi: {e}"})
                    else:
                        st.warning("Không thể khởi tạo dịch vụ Chat. Vui lòng kiểm tra Khóa API.")
            # ****************************** KHUNG CHAT KẾT THÚC ******************************

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
