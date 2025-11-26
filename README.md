# Bank-Churn-Predictor-BigData

Big Data Application for Bank Churn Prediction and Customer Behavior Modeling.

Ứng dụng Big Data trong phân tích hành vi khách hàng và dự đoán rời bỏ

#  1. Mục tiêu của dự án

Dự án này trình bày một giải pháp Big Data End-to-End nhằm giải quyết vấn đề Dự đoán Khách hàng Rời bỏ Ngân hàng (Bank Churn Prediction). Bằng cách tận dụng các kỹ thuật học máy tiên tiến và sức mạnh tính toán trên đám mây (Cloud Computing), chúng tôi phân tích hành vi phức tạp của khách hàng để xác định sớm những người có nguy cơ rời đi, từ đó hỗ trợ ngân hàng đưa ra chiến lược giữ chân khách hàng kịp thời và hiệu quả.

#  2. Các công nghệ được ứng dụng

Python: Ngôn ngữ chính để xử lý dữ liệu và xây dựng mô hình

ML: Xây dựng và tối ưu hóa các mô hình dự đoán rời bỏ

Big Data: Xử lý và phân tích hành vi trên các tập dữ liệu lớn

AWS EC2: Môi trường máy ảo để triển khai và chạy mô hình và Web

#  3. Mô hình và Kết quả Chính

Mô hình dự đoán chính được lựa chọn là mô hình Ensemble (Soft Voting) kết hợp ba thuật toán Random Forest, Gradient Boosting và XGBoost 

Hiệu suất:  F1-score = 0.5053, Precision = 0.6523 và Recall = 0.4123 trên tập dữ liệu thử nghiệm

Phân tích Hành vi: Dự án xác định được các yếu tố hàng đầu dẫn đến sự rời bỏ, bao gồm: Số dư tài khoản hiện tại (curent balance log) và số tiền rút trong tháng này (current month debit log)

<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/4a622863-89c3-4d5c-937d-48b88fc502aa" />

#  4. Quy trình thực hiện tải

Chạy Machine Learning.ipynb để tải mô hình tốt nhất

Khởi tạo EC2: Khởi tạo một instance EC2 (chọn Ubuntu) và SSH trên web AWS

Tải Code web.py, requirements.txt 

Thực thi: Chạy bằng Putty và FileZilla hoặc bất cứ phần mềm nào tương đương

Cài đặt Môi trường: Cài đặt Python 3 và các thư viện cần thiết thông qua pip

Demo là để chạy một list các khách hàng có số liệu bất kỳ
