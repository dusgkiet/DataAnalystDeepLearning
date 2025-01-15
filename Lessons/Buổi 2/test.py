import pandas as pd

# Tạo DataFrame từ Table1
table1_data = {
    "MSSV": [1, 2, 3],
    "HoTen": ["Nguyen", "An", "Binh"],
    "Diem": [7, 4, 8]
}
df1 = pd.DataFrame(table1_data)

# Tạo DataFrame từ Table2
table2_data = {
    "MSSV": [4, 5],
    "HoTen": ["Truc", "Hoa"],
    "Diem": [9, 6]
}
df2 = pd.DataFrame(table2_data)

# Gộp hai DataFrame lại thành một
df_combined = pd.concat([df1, df2], axis=0)
df_combined.reset_index(drop=True, inplace=True) # Reset index

# Thêm cột mới "Đậu" kiểm tra điều kiện "Diem >= 5"
df_combined["Pass"] = df_combined["Diem"] >= 5

# In kết quả
print(df_combined)