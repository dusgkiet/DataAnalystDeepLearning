import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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
df = df.drop('Name', axis=1)

# Hiển thị dữ liệu sau khi tách tên
print("Dữ liệu sau khi tách Firstname và Lastname:")
print(df.head())

# Vấn đề 3: Xử lý không thống nhất đơn vị trong cột Weight
weight = df['Weight']
for i in range(len(weight)):
    x = str(weight[i])
    if "lbs" in x[-3:]:
        x = x[:-3]
        float_x = float(x)
        y = int(float_x / 2.2)
        y = str(y) + "kgs"
        weight[i] = y
df['Weight'] = weight
print("Dữ liệu sau khi chuẩn hóa đơn vị Weight:")
print(df.head())

# Vấn đề 4: Xóa các dòng dữ liệu rỗng
df.dropna(how="all", inplace=True)
plt.figure(figsize=(10, 6))  # Tùy chỉnh kích thước biểu đồ
sns.heatmap(df.isna(),
            yticklabels=False,
            cbar=True,
            cmap='viridis')
plt.show()


# Vấn đề 5: Xóa các dòng dữ liệu trùng lặp dựa trên Firstname, Lastname, Age, Weight
df = df.drop_duplicates(subset=['Firstname', 'Lastname', 'Age', 'Weight'])
# Hiển thị dữ liệu sau khi loại bỏ trùng lặp
print("Dữ liệu sau khi loại bỏ các dòng trùng lặp:")
print(df.head(17))

# Vấn đề 6: Xử lý dữ liệu non-ASCII trong Firstname và Lastname
df.Firstname.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
df.Lastname.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

# Hiển thị dữ liệu sau khi xử lý non-ASCII
print("Dữ liệu sau khi loại bỏ ký tự non-ASCII:")
print(df.head())

# Vấn đề 7: Kiểm tra và xử lý dữ liệu thiếu
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
sns.heatmap(df.isna(),
            yticklabels=False,
            cbar=True,
            cmap='viridis')
plt.show()
index_ageweightnull = df[(df['Age'].isna() & df['Weight'].isna()) & (
    ~df['Id'].isin([4, 5]))].index
print("Các dòng có Age và Weight bị thiếu (trừ ID 4 và 5):",
      index_ageweightnull.tolist())
df.drop(index=index_ageweightnull, inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Weight'] = df['Weight'].fillna(df['Weight'].mean())
print("Dữ liệu sau khi xử lý giá trị thiếu:")
print(df[['Age']].head())

# Vấn đề 8: Tách cột chứa giới tính, thời gian và nhịp tim
df = pd.melt(df, id_vars=['Id', 'Age', 'Weight', 'Firstname', 'Lastname'],
             value_name='PulseRate', var_name='sex_and_time').sort_values(['Id', 'Age', 'Weight', 'Firstname', 'Lastname'])

# Tách Sex và Time từ sex_and_time
tmp_df = df["sex_and_time"].str.extract(r'([mf])(\d{2})(\d{2})', expand=True)
tmp_df.columns = ["Sex", "hours_lower", "hours_upper"]

# Tạo cột Time từ hours_lower và hours_upper
tmp_df["Time"] = tmp_df["hours_lower"] + "-" + tmp_df["hours_upper"]

# Gộp lại với dataframe chính
df = pd.concat([df, tmp_df], axis=1)

# Xóa các cột không cần thiết
df = df.drop(["sex_and_time", "hours_lower", "hours_upper"], axis=1)
df = df.dropna()

print("Dữ liệu sau khi tách cột Sex, Time và PulseRate:")
print(df.head())
df.to_csv("outputcleanup.csv", index=False)

# Vấn đề 11: Xử lý dữ liệu thiếu trên biến huyết áp
def avgPulserate(df, firstname):
    valid_pulses = df.loc[(df['Firstname'] == firstname) & (df['PulseRate'] != 0), 'PulseRate']
    return valid_pulses.mean() if not valid_pulses.empty else 0

def isValidIndex1(index, nrows):
    return index - 1 >= 0 and index + 1 < nrows

def isValidIndex2(index, nrows):
    return index + 2 < nrows

def isPulseRateNotNull1(index, df):
    return df.at[index - 1, 'PulseRate'] != 0 and df.at[index + 1, 'PulseRate'] != 0

def isPulseRateNotNull2(index, df):
    return df.at[index + 1, 'PulseRate'] != 0 and df.at[index + 2, 'PulseRate'] != 0

def isInGroup1(df, index, firstname):
    return df.at[index - 1, 'Firstname'] == firstname and df.at[index + 1, 'Firstname'] == firstname

def isInGroup2(df, index, firstname):
    return df.at[index + 1, 'Firstname'] == firstname and df.at[index + 2, 'Firstname'] == firstname

def updatePulseRateNa(index, df):
    nrows = len(df['PulseRate'])
    firstname = df.at[index, 'Firstname']
    new_value = 0
    try:
        if isValidIndex1(index, nrows) and isPulseRateNotNull1(index, df) and isInGroup1(df, index, firstname):
            new_value = (df.at[index - 1, 'PulseRate'] + df.at[index + 1, 'PulseRate']) / 2
        elif isValidIndex2(index, nrows) and isPulseRateNotNull2(index, df) and isInGroup2(df, index, firstname):
            new_value = (df.at[index + 1, 'PulseRate'] + df.at[index + 2, 'PulseRate']) / 2
        else:
            new_value = avgPulserate(df, firstname)
    except:
        new_value = avgPulserate(df, firstname)
    return new_value

pulse_rate_update = [updatePulseRateNa(i, df) if v == 0 else v for i, v in enumerate(df['PulseRate'])]
df['PulseRate'] = pulse_rate_update