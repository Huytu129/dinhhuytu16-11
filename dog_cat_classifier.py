import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 1. Tiền xử lý dữ liệu
# -------------------------------

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = r"C:\Users\huytu\PycharmProjects\xla\data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "validation")

# Kiểm tra số lượng ảnh trong các thư mục con
train_dog_dir = os.path.join(train_dir, "dog")
train_cat_dir = os.path.join(train_dir, "cat")
val_dog_dir = os.path.join(val_dir, "dog")
val_cat_dir = os.path.join(val_dir, "cat")

print("Train Dog Images:", len(os.listdir(train_dog_dir)))
print("Train Cat Images:", len(os.listdir(train_cat_dir)))
print("Validation Dog Images:", len(os.listdir(val_dog_dir)))
print("Validation Cat Images:", len(os.listdir(val_cat_dir)))

# Tạo các generator để chuẩn bị dữ liệu
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Tạo generator để load ảnh từ thư mục
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary"
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary"
)

# -------------------------------
# 2. Xây dựng mô hình ANN
# -------------------------------

def build_ann():
    model = Sequential([
        Flatten(input_shape=(128, 128, 3)),  # Chuyển ảnh thành vector
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Lớp đầu ra nhị phân (mèo/chó)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# 3. Xây dựng mô hình CNN
# -------------------------------

def build_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Lớp đầu ra nhị phân (mèo/chó)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# 4. Huấn luyện mô hình
# -------------------------------

def train_model(model, train_generator, val_generator, epochs=10):
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )
    return history

# -------------------------------
# 5. Hiển thị kết quả huấn luyện
# -------------------------------

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    plt.figure(figsize=(12, 6))

    # Biểu đồ chính xác (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    # Biểu đồ tổn thất (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

# -------------------------------
# 6. Đánh giá mô hình
# -------------------------------

def evaluate_model(model, val_generator):
    val_generator.reset()  # Đặt lại generator để tránh lỗi khi dự đoán
    val_preds = model.predict(val_generator, verbose=1)
    val_preds = np.round(val_preds).flatten()  # Làm tròn kết quả thành 0 hoặc 1
    print(classification_report(val_generator.classes, val_preds))  # Báo cáo phân loại
    print("Confusion Matrix:")
    print(confusion_matrix(val_generator.classes, val_preds))  # Ma trận nhầm lẫn

# -------------------------------
# 7. Chạy chương trình chính
# -------------------------------

if __name__ == "__main__":
    # Huấn luyện ANN
    print("Training ANN...")
    ann_model = build_ann()
    ann_history = train_model(ann_model, train_generator, val_generator, epochs=10)
    plot_history(ann_history)

    # Huấn luyện CNN
    print("Training CNN...")
    cnn_model = build_cnn()
    cnn_history = train_model(cnn_model, train_generator, val_generator, epochs=10)
    plot_history(cnn_history)

    # Đánh giá mô hình CNN trên tập kiểm tra
    print("Evaluating CNN...")
    evaluate_model(cnn_model, val_generator)
