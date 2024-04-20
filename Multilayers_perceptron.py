
# Import các thư viện cần thiết
import os
import librosa
import numpy as np
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.utils import pad_sequences
from googletrans import Translator
import pyttsx3
from langdetect import detect

# Đường dẫn đến thư mục chứa dữ liệu giọng nói
data_dir = os.path.join(os.path.dirname(__file__), "Voice_training")

# Hàm trích xuất đặc trưng MFCC từ file âm thanh
def extract_features(file_path):
    audio, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)
    mean = np.mean(mfccs, axis=1)
    std = np.std(mfccs, axis=1)
    mfccs = (mfccs.T - mean) / (std + 1e-8)
    return mfccs

# Tạo danh sách đặc trưng và nhãn lớp từ thư mục dữ liệu giọng nói
data = []
labels = []
for label, sub_dir in enumerate(os.listdir(data_dir)):
    dir_path = os.path.join(data_dir, sub_dir)
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        features = extract_features(file_path)
        data.append(features)
        labels.append(label)

# Chuẩn hóa độ dài của các đặc trưng trong danh sách
max_length = max(len(seq) for seq in data)
padded_data = pad_sequences(data, maxlen=max_length, dtype='float32', padding='post', truncating='post')

# Chuyển đổi danh sách thành mảng numpy
labels = np.array(labels)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, random_state=42)

# Chuẩn hóa đặc trưng giọng nói
mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)

# Thay đổi kích thước của đầu vào
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Chuyển đổi nhãn lớp thành one-hot encoding
num_classes = len(np.unique(labels))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Xây dựng mô hình MLP và huấn luyện mô hình
model = Sequential()
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax', input_shape=(X_train.shape[1],)))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, batch_size=64, epochs=25, validation_data=(X_test, y_test))

# Đánh giá hiệu suất
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Hàm để nhận dạng và phiên dịch giọng nói thành văn bản
def predict_speech_to_text(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = r.record(source)
    try:
        text = r.recognize_google(audio_data, language="en-US")
        print("Đoạn văn bản đã nhận dạng: ", text)
    except:
        text = r.recognize_google(audio_data, language='vi')
        print("Đoạn văn bản đã nhận dạng: ", text)
        
    # text = r.recognize_google(audio_data, language='en')
    features = extract_features(file_path)
    normalized_features = (features - mean) / (std + 1e-8)
    normalized_features = pad_sequences([normalized_features], maxlen=max_length, dtype='float32', padding='post', truncating='post')
    predicted_label = model.predict(normalized_features)
    predicted_class = np.argmax(predicted_label[0])
    return text, predicted_class

# Dịch văn bản từ tiếng Anh sang tiếng Việt
def translate_text(text, target_lang='vi'):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    translated_text = translation.text
    return translated_text

# Dự đoán ngôn ngữ từ âm thanh và trả về chỉ số lớp dự đoán
def predict_audio_language(file_path):
    features = extract_features(file_path)
    normalized_features = (features - mean) / (std + 1e-8)
    normalized_features = pad_sequences([normalized_features], maxlen=max_length, dtype='float32', padding='post', truncating='post')
    predicted_label = model.predict(normalized_features)
    predicted_class = np.argmax(predicted_label[0])
    return predicted_class

# Định nghĩa hàm để lấy ngôn ngữ tương ứng
def get_language_label(predicted_class):
    if predicted_class == 0:
        return 'Tiếng Anh'
    elif predicted_class == 1:
        return 'Tiếng Việt'
    else:
        return 'Unknown'

# Hàm chính
def main():
    print("Lựa chọn nguồn dữ liệu:")
    print("1. Sử dụng tệp âm thanh")
    print("2. Nhập văn bản")
    choice = input("Nhập số tương ứng (1 hoặc 2): ")

    if choice == '1':
        audio_file = os.path.join(os.path.dirname(__file__),input("Nhập đường dẫn tệp âm thanh: "))
        predicted_class = predict_audio_language(audio_file)
        predicted_language = get_language_label(predicted_class)
        print("Ngôn ngữ:", predicted_language)

        if predicted_class == 0:
            spoken_text, _ = predict_speech_to_text(audio_file)
            translated_text = translate_text(spoken_text, target_lang='vi')

            print('Văn bản được dịch:', translated_text)

            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            engine.setProperty("voice", voices[1].id)
            engine.say(translated_text)
            engine.runAndWait()

        elif predicted_class == 1:
            spoken_text, _ = predict_speech_to_text(audio_file)
            translated_text = translate_text(spoken_text, target_lang='en')

            print('Văn bản được dịch:', translated_text)

            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            engine.setProperty("voice", voices[0].id)
            engine.say(translated_text)
            engine.runAndWait()

    elif choice == '2':
        text = input("Nhập văn bản cần dịch: ")
        target_lang = input("Nhập mã ngôn ngữ muốn dịch (en: Tiếng Anh, vi: Tiếng Việt): ")
        translated_text = translate_text(text, target_lang)

        print('Văn bản được dịch:', translated_text)

        engine = pyttsx3.init()
        if target_lang == 'en':
            voices = engine.getProperty("voices")
            engine.setProperty("voice", voices[0].id)
        elif target_lang == 'vi':
            voices = engine.getProperty("voices")
            engine.setProperty("voice", voices[1].id)
        engine.say(translated_text)
        engine.runAndWait()

    else:
        print("Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()
