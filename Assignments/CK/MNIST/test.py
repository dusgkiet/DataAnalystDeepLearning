# Import thư viện cần thiết
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
# In ra phiên bản TensorFlow và kiểm tra GPU
print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

# Tải dữ liệu MNIST gồm ảnh chữ số viết tay 0–9
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape dữ liệu ảnh từ (28, 28) → (28, 28, 1) để thêm kênh màu (grayscale)
# 1 kênh màu (grayscale)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Chuyển dữ liệu về kiểu float32 và chuẩn hóa pixel từ 0-255 → 0-1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# (0, 1) là giá trị pixel của ảnh sau khi chuẩn hóa, giúp mô hình học tốt hơn.
# 0 là pixel tối (đen), 1 là pixel sáng (trắng).

# In thông tin kích thước dữ liệu
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Tạo mô hình Sequential gồm các lớp CNN
model = tf.keras.Sequential()
# Sequential là mô hình tuần tự, các lớp được xếp chồng lên nhau theo thứ tự.
# Mỗi lớp sẽ nhận đầu ra của lớp trước làm đầu vào.
# Tạo mô hình CNN với các lớp Conv2D, MaxPooling2D, Flatten, Dense và Dropout.

# Lớp tích chập: 28 kernel (3x3), trích xuất đặc trưng ảnh
model.add(layers.Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))

# Lớp pooling: giảm kích thước ảnh để giảm tính toán
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Chuyển tensor 3D thành vector 1 chiều để đưa vào fully-connected
# các lớp Dense chỉ nhận đầu vào 1 chiều, nên cần flatten trước khi đưa vào.
model.add(layers.Flatten())

# Lớp Dense ẩn với 128 neuron, dùng hàm kích hoạt ReLU
# dùng hàm kích hoạt ReLU để giới thiệu phi tuyến, giúp mô hình học tốt hơn.
model.add(layers.Dense(128, activation=tf.nn.relu))

# Dropout: giảm overfitting bằng cách bỏ ngẫu nhiên 20% neuron
# Điều này giúp mô hình không quá phụ thuộc vào một số neuron cụ thể → tăng khả năng khái quát khi gặp dữ liệu mới.
model.add(layers.Dropout(0.2))

# Lớp đầu ra: 10 lớp tương ứng với 10 chữ số, dùng softmax để phân loại
# Dense(10): tạo lớp có 10 neuron, mỗi neuron tương ứng với 1 chữ số (0–9).
# activation = tf.nn.softmax: biến đầu ra thành xác suất cho từng lớp(tổng 10 giá trị=1), chọn lớp có xác suất cao nhất làm kết quả.
model.add(layers.Dense(10, activation=tf.nn.softmax))

# Biên dịch mô hình với:
# - Adam optimizer
# - sparse_categorical_crossentropy: dùng hàm mất mát cho bài toán phân loại nhiều lớp với nhãn là số nguyên (0–9)
# - Đánh giá bằng độ chính xác accuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình với 20 epoch (lặp 20 lần trên toàn bộ dữ liệu)
model.fit(x=x_train, y=y_train, epochs=20)
# epochs: số lần lặp lại toàn bộ dữ liệu để huấn luyện mô hình.
# Mỗi epoch mô hình sẽ cập nhật trọng số dựa trên dữ liệu huấn luyện.

# evaluate mô hình trên tập test để kiểm tra độ chính xác
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Độ chính xác trên tập test: %.2f%%' % (test_acc * 100))

# Lưu cấu trúc mô hình dưới dạng JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Lưu toàn bộ mô hình (cấu trúc + trọng số) vào file .h5
model.save("model.h5")
print("Saved model to disk")
