import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import base64
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# ==============================
# CẤU HÌNH
# ==============================
MODEL_PATH = "best_model_churn_2.pkl"

LOG_TRANSFORM_COLS = [
    "average_monthly_balance_prevQ", "average_monthly_balance_prevQ2",
    "current_month_credit", "previous_month_credit", "current_month_debit",
    "previous_month_debit", "current_month_balance", "previous_month_balance",
    "current_balance", "previous_month_end_balance"
]

VECTOR_ASSEMBLY_ORDER = [
   'current_balance_log',
   'previous_month_end_balance_log',
   'average_monthly_balance_prevQ_log',
   'previous_month_credit_log',
   'current_month_debit_log',
   'previous_month_debit_log',
   'current_month_balance_log'
]

# ==============================
# HÀM TIỀN XỬ LÝ - GIỐNG HỆT TRONG MODEL
# ==============================
def apply_pyspark_preprocessing(raw_data_dict):
    data = raw_data_dict.copy()
    log_data = {}
    for col in LOG_TRANSFORM_COLS:
        log_data[f"{col}_log"] = np.log1p(data.get(col, 0.0) + 1)
    feature_vector = [log_data.get(col, 0.0) for col in VECTOR_ASSEMBLY_ORDER]
    return np.array(feature_vector).reshape(1, -1)

# Hàm bung vector - GIỐNG HỆT TRONG MODEL
def expand_vector_column(df, col_name):
    """Mở rộng cột vector PySpark thành nhiều cột số Pandas, điền NaN bằng 0.0."""
    # Chuyển đổi SparseVector/DenseVector sang numpy array
    vecs = df[col_name].apply(lambda x: x.toArray() if hasattr(x, "toArray") else np.array(x))

    # Tạo DataFrame từ các mảng/list
    expanded = pd.DataFrame(vecs.tolist(), index=df.index)
    expanded.columns = [f"{col_name}_{i}" for i in range(expanded.shape[1])]

    for col in expanded.columns:
        # Chuyển đổi sang số và điền NaN bằng 0.0
        expanded[col] = pd.to_numeric(expanded[col], errors='coerce').fillna(0.0)

    df = pd.concat([df.drop(columns=[col_name]), expanded], axis=1)
    return df

# Hàm preprocess_final_features - GIỐNG HỆT TRONG MODEL
def preprocess_final_features(df_with_raw_features):
    """
    Nhận df đã có cột 'features_raw', bung vector, điền NaN
    """
    df_processed = expand_vector_column(df_with_raw_features, "features_raw")
    return df_processed.drop(columns=["churn"], errors='ignore').fillna(0)

# Hàm xử lý dữ liệu batch cho file upload
# Hàm xử lý dữ liệu batch cho file upload
def process_batch_data(df):
    """Xử lý batch bằng cách tạo features_raw giống pipeline gốc"""
    try:
        # Tạo DataFrame với cột features_raw cho toàn bộ dữ liệu
        features_list = []
        
        for i, row in df.iterrows():
            try:
                # Tạo input dictionary
                input_dict = {
                    'current_balance': float(row['current_balance']),
                    'previous_month_end_balance': float(row['previous_month_end_balance']),
                    'average_monthly_balance_prevQ': float(row['average_monthly_balance_prevQ']),
                    'previous_month_credit': float(row['previous_month_credit']),
                    'current_month_debit': float(row['current_month_debit']),
                    'previous_month_debit': float(row['previous_month_debit']),
                    'current_month_balance': float(row['current_month_balance']),
                }
                
                # Áp dụng preprocessing
                features_raw_vector = apply_pyspark_preprocessing(input_dict)[0]
                features_list.append(features_raw_vector)
                
            except Exception as e:
                st.error(f"Lỗi khi xử lý dòng {i}: {str(e)}")
                # Thêm vector mặc định nếu có lỗi
                features_list.append(np.zeros(len(VECTOR_ASSEMBLY_ORDER)))
        
        # Tạo DataFrame với cột features_raw
        features_df = pd.DataFrame({
            'features_raw': features_list
        })
        
        # Dùng pipeline để dự đoán toàn bộ
        probabilities = pipeline_dict['pipeline'].predict_proba(features_df)[:, 1]
        return probabilities
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu batch: {str(e)}")
        return np.zeros(len(df))

# ==============================
# TẢI MÔ HÌNH - KHÔNG SỬA PIPELINE
# =============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Không tìm thấy mô hình tại {MODEL_PATH}")
        st.stop()
    
    try:
        # Load model dict - KHÔNG TÁI TẠO PIPELINE
        model_dict = joblib.load(MODEL_PATH)

        
        return model_dict
        
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {str(e)}")
        st.stop()

# Tải model
pipeline_dict = load_model()

# Lấy threshold từ model
MODEL_THRESHOLD = pipeline_dict.get('threshold', 0.757)

# ===============================
# CẤU HÌNH TRANG
# ===============================
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    layout="wide"
)

# ===============================
# CSS TÙY CHỈNH GIAO DIỆN
# ===============================
st.markdown("""
    <style>
        /* Toàn bộ nền trang */
        body {
            background-color: #f8f9fa;
            color: #222222;
            font-family: 'Segoe UI', Roboto, Arial, sans-serif;
        }
        /* Banner */
        .banner {
            background: linear-gradient(to right, #243949, #517fa4);
            color: white;
            padding: 40px 20px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 40px;
        }
        .banner h1 {
            font-size: 36px;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }
        .banner p {
            font-size: 16px;
            color: #e0e0e0;
        }
        /* Phần container nội dung */
        .block {
            background-color: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 25px;
        }
        .stButton>button {
            background-color: #30475e;
            color: white;
            border: none;
            border-radius: 5px;
            height: 45px;
            font-size: 16px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #3c5a7a;
        }
        /* BUTTON MÀU ĐỎ CHO DỰ ĐOÁN HÀNG LOẠT */
        .stButton>button[kind="primary"] {
            background-color: #dc3545 !important;
            border-color: #dc3545 !important;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: #c82333 !important;
            border-color: #bd2130 !important;
        }
        .scenario-button {
            background-color: #f0f2f6 !important;
            color: #30475e !important;
            border: 1px solid #ddd !important;
        }
        .scenario-button:hover {
            background-color: #e4e7eb !important;
        }
        .result-box {
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            margin-top: 20px;
            font-size: 18px;
            font-weight: 500;
            border-left: 5px solid;
        }
        .low-risk {
            background-color: #e9f7ef;
            color: #1e7e34;
            border-left-color: #28a745;
        }
        .high-risk {
            background-color: #f8d7da;
            color: #721c24;
            border-left-color: #dc3545;
        }
        .metric-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin: 10px 0;
        }
        .upload-section {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .success-box {
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #28a745;
            margin: 10px 0;
        }
        .warning-box {
            background-color: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #ffc107;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# BANNER
# ===============================
st.markdown("""
<div class="banner">
    <h1>Customer Churn Prediction</h1>
    <p>Ứng dụng hỗ trợ phân tích và dự đoán khả năng rời bỏ khách hàng dựa trên dữ liệu giao dịch ngân hàng</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# TAB CHỨC NĂNG
# ==============================
tab1, tab2 = st.tabs(["Dự đoán đơn lẻ", "Dự đoán hàng loạt"])

with tab1:
    # ==============================
    # PHẦN NHẬP LIỆU ĐƠN LẺ
    # ==============================
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Nhập thông tin khách hàng")

    with st.form("churn_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            current_balance = st.number_input("Số dư hiện tại (USD)", min_value=0.0, value=5000.0, step=1000.0, format="%.0f")
            previous_month_end_balance = st.number_input("Số dư cuối tháng trước (USD)", min_value=0.0, value=6000.0, step=1000.0, format="%.0f")
            average_monthly_balance_prevQ = st.number_input("Số dư bình quân quý trước (USD)", min_value=0.0, value=5500.0, step=1000.0, format="%.0f")
            previous_month_credit = st.number_input("Tổng tiền nạp tháng trước (USD)", min_value=0.0, value=1000.0, step=1000.0, format="%.0f")
        
        with col2:
            current_month_debit = st.number_input("Tổng tiền rút tháng này (USD)", min_value=0.0, value=4000.0, step=1000.0, format="%.0f")
            previous_month_debit = st.number_input("Tổng tiền rút tháng trước (USD)", min_value=0.0, value=3000.0, step=1000.0, format="%.0f")
            current_month_balance = st.number_input("Số dư trung bình tháng này (USD)", min_value=0.0, value=5000.0, step=1000.0, format="%.0f")
        
        submitted_single = st.form_submit_button("DỰ ĐOÁN VỚI DỮ LIỆU TRÊN", use_container_width=True, type="primary")

    # ==============================
    # DỰ ĐOÁN & HIỂN THỊ KẾT QUẢ ĐƠN LẺ
    # ==============================
    def run_prediction(data, title="Kết quả dự đoán"):
        features = apply_pyspark_preprocessing(data)
        
        try:
            # Tạo DataFrame với features_raw
            features_df = pd.DataFrame({
                'features_raw': [features[0]]
            })
            
            # Dùng pipeline để dự đoán
            prob = pipeline_dict['pipeline'].predict_proba(features_df)[0, 1]
            
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
            prob = 0.0
        
        prediction = 1 if prob >= MODEL_THRESHOLD else 0
        
        st.markdown("---")
        st.subheader(f"{title}")
        
        # Hiển thị metric với styling đẹp
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="margin:0; font-size: 16px;">XÁC SUẤT RỜI BỎ</h3>
            <h1 style="margin:0; font-size: 42px;">{prob*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị kết quả với màu sắc phù hợp
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box high-risk">
                <strong>RỦI RO RỜI BỎ CAO</strong><br>
                <small>Khách hàng có nguy cơ cao rời bỏ - cần chiến lược giữ chân ngay lập tức</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box low-risk">
                <strong>RỦI RO RỜI BỎ THẤP</strong><br>
                <small>Khách hàng có mức độ trung thành tốt - tiếp tục duy trì dịch vụ hiện tại</small>
            </div>
            """, unsafe_allow_html=True)

    # Xử lý khi người dùng bấm nút form đơn lẻ
    if submitted_single:
        input_dict = {
            'current_balance': current_balance,
            'previous_month_end_balance': previous_month_end_balance,
            'average_monthly_balance_prevQ': average_monthly_balance_prevQ,
            'previous_month_credit': previous_month_credit,
            'current_month_debit': current_month_debit,
            'previous_month_debit': previous_month_debit,
            'current_month_balance': current_month_balance,
        }
        run_prediction(input_dict, "Kết quả dự đoán")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # ==============================
    # PHẦN UPLOAD FILE
    # ==============================
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Upload file dữ liệu khách hàng")
    
    st.markdown("""
    <div class="upload-section">
        <h4>Hướng Dẫn Upload File</h4>
        <p>File cần có các cột sau: <code>current_balance</code>, <code>previous_month_end_balance</code>, <code>average_monthly_balance_prevQ</code>, 
        <code>previous_month_credit</code>, <code>current_month_debit</code>, <code>previous_month_debit</code>, <code>current_month_balance</code></p>
        <p>Hỗ trợ định dạng: CSV, Excel</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Chọn file dữ liệu", type=['csv', 'xlsx'], key="batch_upload")
    
    if uploaded_file is not None:
        try:
            # Đọc file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                try:
                    df = pd.read_excel(uploaded_file)
                except ImportError:
                    st.error("""
                    ❌ Thiếu thư viện đọc file Excel. Vui lòng cài đặt bằng lệnh:
                    ```bash
                    pip install openpyxl
                    ```
                    """)
                    st.stop()
            
            # Hiển thị thông tin file
            st.success(f"Đã Upload Thành Công File: {uploaded_file.name}")
            
            # Hiển thị preview dữ liệu
            with st.expander("Xem Trước Dữ Liệu"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Kiểm tra các cột cần thiết
            required_columns = ['current_balance', 'previous_month_end_balance', 'average_monthly_balance_prevQ', 
                              'previous_month_credit', 'current_month_debit', 'previous_month_debit', 'current_month_balance']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Thiếu các cột bắt buộc: {', '.join(missing_columns)}")
            else:
                if st.button("DỰ ĐOÁN HÀNG LOẠT", use_container_width=True, type="primary"):
                    with st.spinner("🔄 Đang xử lý dữ liệu và dự đoán..."):
                        try:
                            # Xử lý dữ liệu batch - hàm mới trả về probabilities trực tiếp
                            probabilities = process_batch_data(df)
                            predictions = (probabilities >= MODEL_THRESHOLD).astype(int)
                            
                            # Thêm kết quả vào dataframe
                            result_df = df.copy()
                            result_df['Xác Suất Rời Bỏ (%)'] = (probabilities * 100).round(2)
                            result_df['Dự Đoán'] = predictions
                            result_df['Trạng Thái'] = result_df['Dự Đoán'].map({0: 'RỦI RO THẤP', 1: 'RỦI RO CAO'})
                            
                            # Đổi tên các cột gốc sang tiếng Việt
                            result_df = result_df.rename(columns={
                                'current_balance': 'Số dư hiện tại (USD)',
                                'previous_month_end_balance': 'Số dư cuối tháng trước (USD)', 
                                'average_monthly_balance_prevQ': 'Số dư bình quân quý trước (USD)',
                                'previous_month_credit': 'Tổng tiền nạp tháng trước (USD)',
                                'current_month_debit': 'Tổng tiền rút tháng này (USD)',
                                'previous_month_debit': 'Tổng tiền rút tháng trước (USD)',
                                'current_month_balance': 'Số dư trung bình tháng này (USD)'
                            })

                            # Hiển thị kết quả
                            st.subheader("Kết quả dự đoán hàng loạt")
                        
                            # Thống kê
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Tổng số khách hàng", len(result_df))
                            with col2:
                                high_risk_count = (result_df['Dự Đoán'] == 1).sum()
                                st.metric("Khách hàng rủi ro cao", high_risk_count)
                            with col3:
                                st.metric("Tỷ lệ khách hàng rời bỏ so với tổng số", f"{(high_risk_count/len(result_df)*100):.1f}%")
                        
                            # Hiển thị bảng kết quả với highlight
                            def highlight_high_risk_rows(row):
                                if row['Trạng Thái'] == 'RỦI RO CAO':
                                    return ['background-color: #ffcccc'] * len(row)
                                return [''] * len(row)

                            styled_df = result_df.style.apply(highlight_high_risk_rows, axis=1)
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Tải về kết quả với HTML button gradient
                            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
                            csv_base64 = base64.b64encode(csv.encode()).decode()
                            
                            st.markdown(f'''
                                <a href="data:file/csv;base64,{csv_base64}" download="ket_qua_du_doan_churn.csv" 
                                   style="display: inline-block; padding: 0.75rem 1.5rem; 
                                          background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); 
                                          color: white; text-decoration: none; border-radius: 0.5rem; 
                                          font-weight: 600; text-align: center; width: 100%; border: none; 
                                          cursor: pointer; transition: all 0.3s ease; margin-top: 1rem;">
                                   Tải kết quả dự đoán (CSV)
                                </a>
                            ''', unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
                        
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div style="text-align:center; margin-top:40px; font-size:13px; color:#777;">
    © 2025 Customer Churn Prediction | Phát triển bởi Nhóm 12
</div>
""", unsafe_allow_html=True)