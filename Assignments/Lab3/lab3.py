import pandas as pd

# Vấn đề 1: Xử lý thiếu header khi đọc file CSV
file_path = "patient_heart_rate.csv"
column_names = ["Id", "Name", "Age", "Weight", "m0006",
                "m0612", "m1218", "f0006", "f0612", "f1218"]
df = pd.read_csv(file_path, names=column_names)

# Hiển thị dữ liệu sau khi thêm header
print("Dữ liệu sau khi thêm header:")
print(df.head())

# Vấn đề 2: Tách cột Name thành Firstname và Lastname
df[['Firstname', 'Lastname']] = df['Name'].str.split(expand=True, n=1)

# Hiển thị dữ liệu sau khi tách tên
print("Dữ liệu sau khi tách Firstname và Lastname:")
print(df.head())
