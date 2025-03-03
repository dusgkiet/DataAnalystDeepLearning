import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Bài 1
print("Bài 1")
def load_data():
    df = pd.read_csv('titanic_disaster.csv')
    print(df.head(10))
    return df

df = load_data()

# Bài 2: Thống kê và heatmap dữ liệu thiếu
print("Bài 2")
print(df.isnull().sum())
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Biểu đồ Heatmap dữ liệu thiếu')
plt.show()

# Bài 3: Tách tên thành lastName, firstName
print("Bài 3")
df[['lastName', 'firstName']] = df['Name'].str.extract(r'([A-Za-z]+),\s(.+)')
df.drop('Name', axis=1, inplace=True)
print(df[['lastName', 'firstName']].head(10))

# Bài 4: Rút gọn dữ liệu cột Sex
print("Bài 4")
df['Sex'] = df['Sex'].replace({'male': 'M', 'female': 'F'})
print(df[['lastName', 'firstName', 'Sex']].head(10))

# Bài 5 
# a
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Phân phối tuổi theo từng hạng vé')
plt.xlabel('Hạng vé (Pclass)')
plt.ylabel('Tuổi (Age)')
plt.grid(True)
plt.show()

# Nhận xét
# Hạng 1: Tuổi trung bình cao hơn hẳn (khoảng trên 35).
# Hạng 2: Tuổi trung bình khoảng 29.
# Hạng 3: Tuổi trung bình thấp nhất, chỉ khoảng 22.
# → Quyết định: Điền dữ liệu thiếu theo trung vị (median) của từng Pclass thay vì toàn bộ dataset.

#b: Điền tuổi theo median của từng hạng vé
def impute_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return df[df['Pclass'] == 1]['Age'].median()
        elif row['Pclass'] == 2:
            return df[df['Pclass'] == 2]['Age'].median()
        else:
            return df[df['Pclass'] == 3]['Age'].median()
    else:
        return row['Age']

df['Age'] = df.apply(impute_age, axis=1)

print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Heatmap dữ liệu thiếu sau khi xử lý Age')
plt.show()

# Bài 6
print("Bài 6")
def age_group(age):
    if age <= 12:
        return 'Kid'
    elif age <= 18:
        return 'Teen'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Older'

df['AgeGroup'] = df['Age'].apply(age_group)

print(df[['Age', 'AgeGroup']].head(10))

# Bài 7
print("Bài 7")
df['NamePrefix'] = df['firstName'].str.extract(r'([A-Za-z]+)\.')

print(df[['firstName', 'NamePrefix']].head(10))

# Bài 8
print("Bài 8")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

print(df[['SibSp', 'Parch', 'FamilySize']].head(10))

# Bài 9
print("Bài 9")
df['Alone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

print(df[['FamilySize', 'Alone']].head(10))

# Bài 10
print("Bài 10")
def extract_cabin_type(cabin):
    if pd.isnull(cabin):
        return 'Unknown'
    else:
        return cabin[0]

df['typeCabin'] = df['Cabin'].apply(extract_cabin_type)

print(df[['Cabin', 'typeCabin']].head(10))

# Bài 11
print("Bài 11")

train, test = train_test_split(df, test_size=0.2, random_state=42)

train = train[~train['PassengerId'].isin(test['PassengerId'])]

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

# Bài 12
plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Survived', data=train, palette='Set2')
plt.title('Tỉ lệ sống sót và thiệt mạng theo giới tính')
plt.show()

# Nhận xét:
# - Phụ nữ (female) có tỉ lệ sống sót cao hơn nam giới rất nhiều.
# - Điều này phù hợp với quy tắc "Phụ nữ và trẻ em lên thuyền cứu sinh trước".

# Bài 13
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=train, palette='Set3')
plt.title('Tỉ lệ sống sót theo hạng vé (Pclass)')
plt.show()

# Nhận xét:
# - Hạng vé cao (Pclass 1) có tỉ lệ sống sót cao nhất.
# - Hạng vé 3 (giá rẻ) có tỉ lệ tử vong cao nhất.

# Bài 14
plt.figure(figsize=(10, 6))
sns.countplot(x='AgeGroup', hue='Survived', data=train, palette='pastel', order=['Kid', 'Teen', 'Adult', 'Older'])
plt.title('Tỉ lệ sống sót theo độ tuổi')
plt.show()

# Nhận xét:
# - Trẻ em (Kid) có tỉ lệ sống sót cao.
# - Người lớn tuổi (Older) tỉ lệ sống sót thấp nhất.

# Bài 15
plt.figure(figsize=(10, 5))
sns.barplot(x='FamilySize', y='Survived', data=train, palette='Blues')
plt.title('Xác suất sống sót theo kích thước gia đình')
plt.show()

# Nhận xét:
# - Người đi 1 mình (Alone) có tỉ lệ sống sót thấp.
# - Người đi theo nhóm nhỏ (2-4 người) có tỉ lệ sống sót cao nhất.
# - Gia đình quá đông (trên 5 người) tỉ lệ sống sót lại giảm.

# Bài 16
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Fare', data=train, palette='coolwarm')
plt.title('Xác suất sống sót theo giá vé')
plt.show()

# Nhận xét:
# - Giá vé càng cao thì khả năng sống sót càng cao.
# - Điều này liên quan tới việc hạng vé cao (Pclass 1) thường đắt tiền và ưu tiên cứu trước.

# Bài 17
plt.figure(figsize=(10, 6))
sns.catplot(x='Embarked', hue='Survived', col='Pclass', data=train, kind='count', palette='Set1')
plt.suptitle('Tỉ lệ sống sót theo cảng lên tàu và hạng vé')
plt.show()

# Nhận xét:
# - Cảng C (Cherbourg) có tỉ lệ sống sót cao nhất, đặc biệt ở Pclass 1.
# - Cảng S (Southampton) có nhiều hành khách hạng 3 nhất và tỉ lệ tử vong cao.