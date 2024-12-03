import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Veri seti dizinleri
original_dataset_dir = "./dataset"
base_dir = "./split_dataset"

# Eğitim, doğrulama ve test klasörlerini oluşturma
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# Ana klasörler oluşturuluyor
if not os.path.exists(base_dir):
    os.makedirs(train_dir)
    os.makedirs(validation_dir)
    os.makedirs(test_dir)

    # Her bir sınıf için alt klasörler oluşturuluyor ve görüntüler bölünüyor
    for class_name in os.listdir(original_dataset_dir):
        class_path = os.path.join(original_dataset_dir, class_name)
        if os.path.isdir(class_path):
            print(f"{class_name} sınıfı işleniyor...")

            # Alt klasörler oluşturuluyor
            train_class_dir = os.path.join(train_dir, class_name)
            validation_class_dir = os.path.join(validation_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(validation_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Görüntü dosyalarını al
            images = os.listdir(class_path)
            images = [img for img in images if os.path.isfile(os.path.join(class_path, img))]

            # Veriyi %70 eğitim, %15 doğrulama, %15 test olarak böl
            train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
            validation_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

            # Görüntüleri uygun klasörlere taşı
            for img in train_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
            for img in validation_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(validation_class_dir, img))
            for img in test_images:
                shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

    print("Veri seti başarıyla bölündü.")
else:
    print("Veri seti zaten bölünmüş.")

# Veri artırma ve veri seti oluşturma
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Transfer öğrenme modeli (ResNet50)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Model yapısı
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  # Sınıf sayısına göre çıktı
])

# Modeli derleme
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model kontrol noktası ayarı
checkpoint_path = "model_checkpoint.keras"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)


# Erken durdurma
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Modeli eğitme veya yükleme
if os.path.exists(checkpoint_path):
    print("Kaydedilmiş model yükleniyor...")
    model = load_model(checkpoint_path)
else:
    print("Yeni model eğitiliyor...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=[early_stopping, checkpoint]
    )

# Performansı değerlendirme
loss, accuracy = model.evaluate(test_generator)
print(f"Test Doğruluğu: {accuracy * 100:.2f}%")

# Eğitim ve doğrulama doğruluklarını görselleştirme
if 'history' in locals():  # Eğer eğitim yapıldıysa
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.legend()
    plt.title('Model Doğrulukları')
    plt.xlabel('Epochs')
    plt.ylabel('Doğruluk')
    plt.show()

    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.legend()
    plt.title('Model Kayıpları')
    plt.xlabel('Epochs')
    plt.ylabel('Kayıp')
    plt.show()
