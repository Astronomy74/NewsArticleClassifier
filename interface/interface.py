import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QPlainTextEdit
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt
import pickle
import re
import num2words
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        def clean_text(web_text):
            # lowercasing
            text_clean = web_text.lower()
            # converting numbers to words
            text_words = text_clean.split()
            converted_words = []
            for word in text_words:
                try:
                    converted_word = num2words.num2words(int(word))
                    converted_words.append(converted_word)
                except ValueError:
                    converted_words.append(word)
            text_clean = ' '.join(converted_words)
            # removing special characters and numbers
            text_clean = re.sub(r'[^a-z]', ' ', text_clean)
            # removing stop words
            stop_words = set(nltk.corpus.stopwords.words("english"))
            text_clean = ' '.join([word for word in text_clean.split() if word not in stop_words])
            # lemmatization
            wordnet_lemmatizer = WordNetLemmatizer()
            lemmatized_list = []
            text_words = text_clean.split(" ")
            for word in text_words:
                lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
            text_clean = " ".join(lemmatized_list)
            # stemming
            stemmer = PorterStemmer()
            text_clean = stemmer.stem(text_clean)
            return text_clean

        # Set the window title
        self.setWindowTitle("News Article Classifier")

        # Set the window size
        self.setGeometry(350, 200, 1197, 636)

        # Load the background image
        background_image = QPixmap("C:/Users/satur/Desktop/ANN project/interface/background.png")

        
        background_label = QLabel(self)
        background_label.setPixmap(background_image)
        background_label.setGeometry(0, 0, 1197, 636)

        
        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setGeometry(20, 215, 450, 350)

        font = QFont("Calisto MT", 15)
        font.setBold(True)
        font.setItalic(True)
        self.text_edit.setFont(font)
       
        self.text_edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        
        self.text_edit.setStyleSheet("background-color: transparent; color: #363333;")


        
        button = QPushButton(self)
        button.setText("          ")
        button.setGeometry(190, 570, 90, 30)
        button.setStyleSheet("background-color: transparent; color: #FFFFFF; border: none;")
        button.setFlat(True)  

        model_file = "C:/Users/satur/Desktop/ANN project/saved model/nac_model.pkl"
        vectorizer_file = "C:/Users/satur/Desktop/ANN project/saved model/tfidf_vectorizer.pkl"
        
        label = QLabel(self)
        label.setGeometry(660, 490, 380, 100)

        # Set the font properties
        lableFont = QFont("Calisto MT", 30)
        lableFont.setBold(True)
        label.setFont(lableFont)
        # Set the alignment to center horizontally and vertically
        label.setAlignment(Qt.AlignCenter)
        # Set the stylesheet for the QLabel widget to make it transparent
        label.setStyleSheet("background-color: transparent; color: #363333")

        def button_pressed():
            text = self.text_edit.toPlainText() 
            with open(model_file, 'rb') as file:
                loaded_model = pickle.load(file)
            with open(vectorizer_file, 'rb') as file:
                vectorizer = pickle.load(file)

            preprocessed_text = clean_text(text)
            text_vector = vectorizer.transform([preprocessed_text])

            category_prediction = loaded_model.predict(text_vector)[0]
            print("Predicted category:", category_prediction)

            label.setText(category_prediction)
            
            

        def button_released():
            pass

        button.setStyleSheet(
        """
        QPushButton {
            background-color: transparent;
            color: #FFFFFF;
            border: none;
        }
        QPushButton:pressed {
            background-color: rgba(128, 128, 128, 100);
            color: #FFFFFF;
        }
        QPushButton:released {
            background-color: transparent;
            color: #FFFFFF;
        }
        """
        )

        button.pressed.connect(button_pressed)
        button.released.connect(button_released)


if __name__ == "__main__":
    model_file = "C:/Users/satur/Desktop/ANN project/saved model/nac_model.pkl"
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
    icon_path = "C:/Users/satur/Desktop/ANN project/interface/news.ico"
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(icon_path))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


