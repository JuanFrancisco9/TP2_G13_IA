import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# Parámetros para MFCC
SAMPLE_RATE = 22050  # Hz
DURATION = 7         # segundos (truncar o padding)
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 13          # número de coeficientes MFCC

def extract_mfcc(file_path, max_pad_len=130):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        # Padding o truncado para tener todos del mismo tamaño
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print(f"Error al procesar {file_path}: {e}")
        return None

def load_data(data_dir):
    X = []
    y = []
    labels = {'not_cry': 0, 'cry': 1}

    for label in labels:
        class_dir = os.path.join(data_dir, label)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav') or filename.endswith('.ogg') :
                file_path = os.path.join(class_dir, filename)
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    X.append(mfcc)
                    y.append(labels[label])

    X = np.array(X)
    y = np.array(y)
    return X, y

# Cargar datasets
X_train, y_train = load_data('data/train')
X_val, y_val = load_data('data/validation')
X_test, y_test = load_data('data/test')

# Redimensionar para Keras: (samples, height, width, channels)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Codificar etiquetas
y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat = to_categorical(y_val, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# Modelo simple
model = Sequential([
    Flatten(input_shape=X_train.shape[1:]),
    Dense(10, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,
    batch_size=32,
    #callbacks=[early_stop]
)

# Función para mostrar matriz de confusión
def mostrar_matriz_confusion(X, y_true, dataset_name):
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['not_cry', 'cry'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Matriz de Confusión - {dataset_name}')
    plt.show()

    # Reporte de métricas
    print(f'\nReporte de Clasificación - {dataset_name}')
    print(classification_report(y_true, y_pred_classes, target_names=['not_cry', 'cry']))

# Mostrar matriz para datos de entrenamiento
mostrar_matriz_confusion(X_train, y_train, 'Entrenamiento')

# Mostrar matriz para datos de prueba
mostrar_matriz_confusion(X_test, y_test, 'Prueba')

# Gráfico del error durante el entrenamiento
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Loss Entrenamiento')
plt.plot(history.history['val_loss'], label='Loss Validación')
plt.title('Progreso del Error (Loss) durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluación
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f'Precisión en conjunto de prueba: {test_acc:.2f}')