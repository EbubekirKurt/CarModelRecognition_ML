import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


def main():
    # Veri setinin yolu ve görüntü boyutları
    dataset_path = './dataset'
    img_width, img_height = 224, 224

    # Veri setini yüklüyoruz
    print("Veri seti yükleniyor...")
    images = []
    labels = []
    # Tüm sınıfları alıyoruz
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    for class_name in class_dirs:
        class_dir = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Görüntüyü yüklüyor ve istenilen boyuta getiriyoruz
                img = load_img(img_path, target_size=(img_width, img_height))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_name)
            except Exception as e:
                print(f"Hata oluştu: {e}")

    images = np.array(images)
    labels = np.array(labels)

    # Verileri eğitim ve test olarak bölüyoruz
    X_train_img, X_test_img, y_train_labels, y_test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels)

    # Etiketleri sayısal değerlere dönüştürüyoruz
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_labels)
    y_test_encoded = label_encoder.transform(y_test_labels)

    # ResNet50 modelini önceden eğitilmiş ağırlıklarla yüklüyoruz
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    preprocess_input = resnet_preprocess

    # Görüntüleri ön işleme tabi tutuyoruz
    X_train_preprocessed = preprocess_input(X_train_img)
    X_test_preprocessed = preprocess_input(X_test_img)

    # Özellikleri çıkarıyorum
    print("\nÖzellikler çıkarılıyor...")
    X_train_features = base_model.predict(X_train_preprocessed)
    X_test_features = base_model.predict(X_test_preprocessed)

    # Özellikleri ölçeklendiriyoruz
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)

    # Modelleri eğitiyor ve değerlendiriyoruz
    classifiers = {
        'SVM': SVC(kernel='linear'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    model_results = {}
    for clf_name, clf in classifiers.items():
        print(f"\n{clf_name} modeli eğitiliyor...")
        clf.fit(X_train_scaled, y_train_encoded)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test_encoded, y_pred)
        cm = confusion_matrix(y_test_encoded, y_pred)
        print(f"{clf_name} Doğruluğu: {acc * 100:.2f}%")
        model_results[clf_name] = {
            'accuracy': acc,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }

    # En iyi modeli seçiyoruz
    best_model_name = max(model_results, key=lambda x: model_results[x]['accuracy'])
    best_model = model_results[best_model_name]

    # Sonuçları yazdırıyoruz
    print(f"\nEn iyi algoritma: {best_model_name}")
    print(f"En iyi doğruluk: {best_model['accuracy'] * 100:.2f}%")

    # Sınıflandırma raporunu yazdırıyoruz
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test_encoded, best_model['y_pred'],
          target_names=label_encoder.classes_))

    # Confusion Matrix'i görselleştiriyoruz
    cm = best_model['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{best_model_name} Confusion Matrix')
    plt.xticks(rotation=90)
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
