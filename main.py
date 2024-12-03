import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


def load_dataset(path, img_width=224, img_height=224, selected_classes=None):
    images = []
    labels = []
    class_names = []
    if selected_classes is None:
        # Tüm sınıfları liste olarak alıyoruz
        selected_classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for class_name in selected_classes:
        class_dir = os.path.join(path, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_names.append(class_name)
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
    return np.array(images), np.array(labels), class_names


def preprocess_data(images, labels, model_name='ResNet50'):
    if model_name == 'ResNet50':
        preprocess_input = resnet_preprocess
    elif model_name == 'VGG16':
        preprocess_input = vgg_preprocess
    else:
        raise ValueError("Model adı 'ResNet50' veya 'VGG16' olmalıdır.")

    # Görüntüleri ön işleme tabi tutuyoruz
    images = preprocess_input(images)
    # Etiketleri sayısal değerlere dönüştürüyoruz
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    return images, labels_encoded, label_encoder


def extract_features(images, model_name='ResNet50'):
    if model_name == 'ResNet50':
        # ResNet50 modelini önceden eğitilmiş ağırlıklarla yüklüyoruz
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == 'VGG16':
        # VGG16 modelini önceden eğitilmiş ağırlıklarla yüklüyoruz
        base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    else:
        raise ValueError("Model adı 'ResNet50' veya 'VGG16' olmalıdır.")
    # Özellikleri çıkarıyoruz
    features = base_model.predict(images, verbose=1)
    return features


def evaluate_models(X_train, X_test, y_train, y_test, class_names):
    classifiers = {
        'SVM': SVC(kernel='linear', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    model_results = {}
    for clf_name, clf in classifiers.items():
        print(f"\n{clf_name} modeli eğitiliyor...")
        # Modeli eğitiyoruz
        clf.fit(X_train, y_train)
        # Tahminleri yapıyoruz
        y_pred = clf.predict(X_test)
        # Doğruluğu hesaplıyoruz
        acc = accuracy_score(y_test, y_pred)
        # Confusion Matrix oluşturuyoruz
        cm = confusion_matrix(y_test, y_pred)
        print(f"{clf_name} Doğruluğu: {acc * 100:.2f}%")
        model_results[clf_name] = {
            'accuracy': acc,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }
    # En iyi modeli seçiyoruz
    best_model_name = max(model_results, key=lambda x: model_results[x]['accuracy'])
    best_model = model_results[best_model_name]
    return model_results, best_model_name, best_model


def main():
    dataset_path = './dataset'  # Veri setinin yolu
    img_width, img_height = 224, 224

    # Veri setini yüklüyoruz (tüm sınıflar)
    print("Veri seti yükleniyor...")
    images, labels, class_names = load_dataset(dataset_path, img_width, img_height)

    # Verileri eğitim ve test olarak bölüyoruz
    X_train_img, X_test_img, y_train_labels, y_test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels)

    # Veri ön işleme ve özellik çıkarma için kullanılacak modeli seçiyoruz
    model_name = 'ResNet50'  # veya 'VGG16'

    # Eğitim verilerini ön işliyor ve özellik çıkarıyoruz
    print(f"\n{model_name} ile eğitim verilerinden özellikler çıkarılıyor...")
    X_train_preprocessed, y_train_encoded, label_encoder = preprocess_data(
        X_train_img, y_train_labels, model_name)
    X_train_features = extract_features(X_train_preprocessed, model_name)

    # Test verilerini ön işliyor ve özellik çıkarıyoruz
    print(f"\n{model_name} ile test verilerinden özellikler çıkarılıyor...")
    X_test_preprocessed, y_test_encoded, _ = preprocess_data(
        X_test_img, y_test_labels, model_name)
    X_test_features = extract_features(X_test_preprocessed, model_name)

    # Özellikleri ölçeklendiriyoruz
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)

    # Modelleri eğitiyor ve değerlendiriyoruz
    model_results, best_model_name, best_model = evaluate_models(
        X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, class_names)

    # En iyi modelin sonuçlarını yazdırıyoruz
    print(f"\nEn iyi algoritma: {best_model_name}")
    print(f"En iyi doğruluk: {best_model['accuracy'] * 100:.2f}%")
    print("En iyi Confusion Matrix:")
    print(best_model['confusion_matrix'])

    # Sınıflandırma raporunu yazdırıyoruz
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test_encoded, best_model['y_pred'],
          target_names=label_encoder.classes_))

    # Confusion Matrix'i görselleştiriyoruz
    cm = confusion_matrix(y_test_encoded, best_model['y_pred'])
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
