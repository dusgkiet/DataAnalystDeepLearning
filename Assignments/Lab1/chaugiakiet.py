import pandas as pd

# Câu 1: Tải dữ liệu từ file CSV
data = pd.read_csv('dulieuxettuyendaihoc.csv')

# Câu 2: Phân loại dữ liệu định tính và định lượng
qualitative_columns = ['GT', 'DT', 'KV', 'KT']
quantitative_columns = [col for col in data.columns if col not in qualitative_columns + ['STT']]

# Câu 3: In ra 10 dòng đầu và 10 dòng cuối
print("\nCâu 3:")
print("10 dòng đầu:")
print(data.head(10))
print("\n10 dòng cuối:")
print(data.iloc[-10:])  # Thay thế data.tail(10) bằng data.iloc[-10:] để đảm bảo lấy đúng từ 91 đến 100

# Câu 4: Thống kê dữ liệu thiếu cho cột DT và thay thế dữ liệu thiếu bằng 0
print("\nCâu 4:")
print("\nThống kê dữ liệu thiếu cho cột DT:")
print(data['DT'].value_counts(dropna=False))
data['DT'] = data['DT'].fillna(0)
print("\nSau khi thay thế dữ liệu thiếu bằng 0:")
print(data['DT'].value_counts())

# Câu 5: Thống kê dữ liệu thiếu cho T1 và thay thế bằng giá trị trung bình
print("\nCâu 5:")
print("\nThống kê dữ liệu thiếu cho cột T1 trước khi xử lý:")
print(data['T1'].isnull().sum())
mean_T1 = data['T1'].mean()
data['T1'] = data['T1'].fillna(mean_T1)
print("\nSau khi thay thế dữ liệu thiếu bằng giá trị trung bình:")
print(data['T1'].isnull().sum())

# Câu 6: Xử lý dữ liệu thiếu cho các cột điểm số còn lại bằng giá trị trung bình
print("\nCâu 6:")
for col in quantitative_columns:
    if data[col].isnull().sum() > 0:
        mean_value = data[col].mean()
        data[col].fillna(mean_value, inplace=True)

# Câu 7: Tạo các biến TBM1, TBM2, TBM3
print("\nCâu 7:")
data['TBM1'] = (data['T1']*2 + data['L1'] + data['H1'] + data['S1'] + data['V1']*2 + data['X1'] + data['D1'] + data['N1']) / 10
data['TBM2'] = (data['T2']*2 + data['L2'] + data['H2'] + data['S2'] + data['V2']*2 + data['X2'] + data['D2'] + data['N2']) / 10
data['TBM3'] = (data['T6']*2 + data['L6'] + data['H6'] + data['S6'] + data['V6']*2 + data['X6'] + data['D6'] + data['N6']) / 10
print(data[['TBM1', 'TBM2', 'TBM3']])

# Câu 8: Tạo các biến xếp loại XL1, XL2, XL3
print("\nCâu 8:")
def classify_grade(tbm):
    if tbm < 5.0:
        return 'Y'
    elif 5.0 <= tbm < 6.5:
        return 'TB'
    elif 6.5 <= tbm < 8.0:
        return 'K'
    elif 8.0 <= tbm < 9.0:
        return 'G'
    else:
        return 'XS'

data['XL1'] = data['TBM1'].apply(classify_grade)
data['XL2'] = data['TBM2'].apply(classify_grade)
data['XL3'] = data['TBM3'].apply(classify_grade)

print(data[['XL1', 'XL2', 'XL3']])

# Câu 9: Tạo các biến US_TBM1, US_TBM2, US_TBM3 theo thang điểm 4 của Mỹ
print("\nCâu 9:")
def min_max_normalization(value, min_val=0, max_val=10, target_min=0, target_max=4):
    return (value - min_val) / (max_val - min_val) * (target_max - target_min) + target_min

data['US_TBM1'] = data['TBM1'].apply(min_max_normalization)
data['US_TBM2'] = data['TBM2'].apply(min_max_normalization)
data['US_TBM3'] = data['TBM3'].apply(min_max_normalization)

print(data[['US_TBM1', 'US_TBM2', 'US_TBM3']])

# Câu 10: Tạo biến KQXT để xác định sinh viên đậu hoặc rớt
print("\nCâu 10:")
def determine_admission_result(dh1, dh2, dh3, kt):
    if kt in ['A', 'A1']:
        score = (dh1 * 2 + dh2 + dh3) / 4
    elif kt == 'B':
        score = (dh1 + dh2 * 2 + dh3) / 4
    else:
        score = (dh1 + dh2 + dh3) / 3
    return 1 if score >= 5.0 else 0

data['KQXT'] = data.apply(lambda row: determine_admission_result(row['DH1'], row['DH2'], row['DH3'], row['KT']), axis=1)

print(data['KQXT'])

# Câu 11: Lưu dữ liệu đã xử lý xuống file CSV
print("\nCâu 11:")
output_file = 'chaugiakiet_dulieuxettuyendaihoc.csv'
data.to_csv(output_file, index=False)
print(f"Dữ liệu đã được lưu vào file {output_file}")