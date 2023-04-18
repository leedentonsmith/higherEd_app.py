import sys
import openai
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QPushButton, QLabel, QInputDialog,
                             QComboBox, QDialog, QFileDialog, QSlider, QTabWidget, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt
import requests
from bs4 import BeautifulSoup
from transformers import MarianMTModel, AutoTokenizer
from sacrebleu import Tokenizer
import language_tool_python
from transformers import (AutoModelForSequenceClassification, AutoModelForTokenClassification,
                          AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, TFAutoModelForSeq2SeqLM,
                          AutoTokenizer, TextClassificationPipeline, TokenClassificationPipeline,
                          QuestionAnsweringPipeline, TranslationPipeline)


model_name = "Helsinki-NLP/opus-mt-en-fr"

import sys
import os

print("Python path:", sys.path)
print("Python executable:", sys.executable)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
sacrebleu_tokenizer = Tokenizer("13a")  # '13a' is the default tokenizer in sacrebleu

def translate(text, model, tokenizer, sacrebleu_tokenizer):
    inputs = tokenizer.encode(text, return_tensors="pt", padding=True)
    outputs = model.generate(inputs, num_beams=4, max_length=512)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sacrebleu_tokenizer.detokenize(decoded.split(" "))

# Your translation function call would look like this:
translated_text = translate("Hello, world!", model, tokenizer, sacrebleu_tokenizer)

tool = language_tool_python.LanguageTool("en-US")

def check_grammar_spelling(text):
    matches = tool.check(text)
    corrected_text = tool.correct(text)
    return matches, corrected_text

errors, corrected_text = check_grammar_spelling("Your input text here")

# NER
ner_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner_pipeline = TokenClassificationPipeline(model=ner_model, tokenizer=ner_tokenizer)

# Zero-shot classification
classification_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
classification_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
classification_pipeline = TextClassificationPipeline(model=classification_model, tokenizer=classification_tokenizer)

# Question-answering
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qa_pipeline = QuestionAnsweringPipeline(model=qa_model, tokenizer=qa_tokenizer)

# Translation
translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
translation_pipeline = TranslationPipeline(model=translation_model, tokenizer=translation_tokenizer)

def fetch_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return ' '.join(soup.stripped_strings)

class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.init_nlp()
        
        # Create summarization_depth_slider and process_url_button
        self.summarization_depth_slider = QSlider(Qt.Horizontal)
        self.summarization_depth_slider.setMinimum(10)
        self.summarization_depth_slider.setMaximum(500)
        self.summarization_depth_slider.setValue(50)
        
        self.process_url_button = QPushButton("Process URL")

        # Create a layout and add widgets to it
        layout = QVBoxLayout()
        layout.addWidget(self.summarization_depth_slider)
        layout.addWidget(self.process_url_button)

        # Set the layout for the central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.process_url_button.clicked.connect(self.process_url)

    def initUI(self):
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()

        self.apply_stylesheet()

        self.tab_summarize = QWidget()
        self.tab_sentiment = QWidget()
        self.tab_generate = QWidget()

        self.tabs.addTab(self.tab_summarize, "Summarize")
        self.tabs.addTab(self.tab_sentiment, "Analyze Sentiment")
        self.tabs.addTab(self.tab_generate, "Generate Text")

        layout.addWidget(self.tabs)

        self.create_summarize_tab()
        self.create_sentiment_tab()
        self.create_generate_tab()
        
        self.settings_button = QPushButton("Settings")
        self.save_button = QPushButton("Save")
        self.import_button = QPushButton("Import")

        layout.addWidget(self.settings_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.import_button)

        self.setCentralWidget(central_widget)
        self.setWindowTitle("Text Processing App")
        self.setGeometry(100, 100, 600, 400)

        self.settings_button.setToolTip("Open settings to adjust summarization depth, creativity, and max length.")
        self.save_button.setToolTip("Save the output text to a file.")
        self.import_button.setToolTip("Import text from a file.")


        self.settings_button.clicked.connect(self.open_settings)
        self.save_button.clicked.connect(self.save_output)
        self.import_button.clicked.connect(self.import_text)

    def init_nlp(self):
        # Sentiment Analysis
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_pipeline = TextClassificationPipeline(model=sentiment_model, tokenizer=sentiment_tokenizer)

        # Summarization
        summarization_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
        summarization_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        self.summarization_pipeline = TranslationPipeline(model=summarization_model, tokenizer=summarization_tokenizer)

        # Text Generation
        text_generation_model = AutoModelForSequenceClassification.from_pretrained("gpt2")
        text_generation_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.text_generation_pipeline = TextGenerationPipeline(model=text_generation_model, tokenizer=text_generation_tokenizer)

        openai.api_key = "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"

    def create_generate_tab(self):
        layout = QVBoxLayout(self.tab_generate)

        self.text_input_generate = QTextEdit()
        self.process_generate_button = QPushButton("Generate Text")
        self.text_output_generate = QTextEdit()

        layout.addWidget(self.text_input_generate)
        layout.addWidget(self.process_generate_button)
        layout.addWidget(self.text_output_generate)

        self.process_generate_button.clicked.connect(self.generate_text)

    def create_summarize_tab(self):
        layout = QVBoxLayout(self.tab_summarize)

        self.text_input = QTextEdit()
        self.process_summarize_button = QPushButton("Summarize Text")
        self.text_output = QTextEdit()

        layout.addWidget(self.text_input)
        layout.addWidget(self.process_summarize_button)
        layout.addWidget(self.text_output)

        self.process_summarize_button.clicked.connect(self.summarize_text)

    def create_sentiment_tab(self):
        layout = QVBoxLayout(self.tab_sentiment)

        self.text_input_sentiment = QTextEdit()
        self.process_sentiment_button = QPushButton("Analyze Sentiment")
        self.text_output_sentiment = QTextEdit()

        layout.addWidget(self.text_input_sentiment)
        layout.addWidget(self.process_sentiment_button)
        layout.addWidget(self.text_output_sentiment)

        self.process_sentiment_button.clicked.connect(self.analyze_sentiment)


    def open_settings(self):
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("Settings")

        settings_layout = QVBoxLayout(settings_dialog)

        summarization_depth_label = QLabel("Summarization Depth:")
        settings_layout.addWidget(summarization_depth_label)
        settings_layout.addWidget(self.summarization_depth_slider)

        creativity_label = QLabel("Creativity:")
        settings_layout.addWidget(creativity_label)

        self.creativity_slider = QSlider(Qt.Horizontal)
        self.creativity_slider.setMinimum(0)
        self.creativity_slider.setMaximum(100)
        self.creativity_slider.setValue(50)
        settings_layout.addWidget(self.creativity_slider)

        max_length_label = QLabel("Max Length:")
        settings_layout.addWidget(max_length_label)

        self.max_length_slider = QSlider(Qt.Horizontal)
        self.max_length_slider.setMinimum(10)
        self.max_length_slider.setMaximum(500)
        self.max_length_slider.setValue(100)
        settings_layout.addWidget(self.max_length_slider)

        settings_dialog.setLayout(settings_layout)

        if settings_dialog.exec_():
            self.creativity_slider.setValue(self.creativity_slider.value())
            self.max_length_slider.setValue(self.max_length_slider.value())

    def save_output(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt);;All Files (*)")

        if file_name:
            with open(file_name, "w") as file:
                file.write(self.text_output.toPlainText())

    def summarize_text(self):
        input_text = self.text_input.toPlainText()
        max_length = self.max_length_slider.value()
        min_length = self.summarization_depth_slider.value()
        summary = self.summarization_pipeline(input_text, max_length=max_length, num_return_sequences=1)
        self.text_output.setPlainText(summary[0]['summary_text'])

    def analyze_sentiment(self):
        input_text = self.text_input_sentiment.toPlainText()
        sentiment = self.sentiment_pipeline(input_text)
        sentiment_result = sentiment[0]['label'].lower()
        sentiment_score = round(sentiment[0]['score'], 2)
    
        prompt = f"Provide suggestions on how to improve the sentiment of the text '{input_text}' with a current sentiment score of {sentiment_score}."
        suggestions = self.text_generation_pipeline(prompt, max_length=50, num_return_sequences=1)
        suggestion_text = suggestions[0]['generated_text'].replace(prompt, "").strip()
    
        self.text_output_sentiment.setPlainText(f"Sentiment: {sentiment_result}, Score: {sentiment_score}\n\nSuggestions:\n{suggestion_text}")

    def generate_text(self):
        input_text = self.text_input_generate.toPlainText()
        generated_text = self.generate_text_openai(input_text)
        self.text_output_generate.setPlainText(generated_text)

    def generate_text_openai(self, prompt):
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=self.creativity_slider.value() / 100.0,
            top_p=1
        )

        return response.choices[0].text.strip()

        
    def process_url(self):
        url, _ = QInputDialog.getText(self, "Enter URL", "URL:")
        if url:
            html_content = fetch_url_content(url)
            if html_content:
                text_content = extract_text_from_html(html_content)
                current_tab = self.tabs.currentIndex()

                if current_tab == 0:  # Summarize tab
                    self.text_input.setPlainText(text_content)
                    self.summarize_text()
                elif current_tab == 1:  # Sentiment tab
                    self.text_input_sentiment.setPlainText(text_content)
                    self.analyze_sentiment()
                elif current_tab == 2:  # Generate tab
                    self.text_input_generate.setPlainText(text_content)
                    self.generate_text()

    def save_output(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt);;All Files (*)")

        if file_name:
            with open(file_name, "w") as file:
                current_tab = self.tabs.currentIndex()

                if current_tab == 0:  # Summarize tab
                    file.write(self.text_output.toPlainText())
                elif current_tab == 1:  # Sentiment tab
                    file.write(self.text_output_sentiment.toPlainText())
                elif current_tab == 2:  # Generate tab
                    file.write(self.text_output_generate.toPlainText())
                    
    def import_text(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt);;All Files (*)")

        if file_name:
            with open(file_name, "r") as file:
                self.text_input.setPlainText(file.read())

    def apply_stylesheet(self):
        stylesheet = """
        /* Set background color and font for the app */
        QWidget {
            background-color: #f0f0f0;
            font-family: "Segoe UI";
            font-size: 11pt;
        }

        /* Set background color and font for buttons */
        QPushButton {
            background-color: #007BFF;
            color: #ffffff;
            border: none;
            padding: 5px;
            border-radius: 3px;
        }

        QPushButton:hover {
            background-color: #0056b3;
        }

        QPushButton:pressed {
            background-color: #004085;
        }

        /* Set background color and font for QTextEdit */
        QTextEdit {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #c0c0c0;
            border-radius: 3px;
        }

        /* Set background color and font for QSlider */
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 6px;
            background: #f0f0f0;
            margin: 2px 0;
        }

        QSlider::handle:horizontal {
            background: #007BFF;
            border: none;
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }

        /* Set background color and font for QTabWidget */
        QTabWidget::pane {
            border: 1px solid #999999;
            background-color: #f0f0f0;
        }

        QTabBar::tab {
            background-color: #c0c0c0;
            padding: 5px;
            border: none;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
        }

        QTabBar::tab:selected {
            background-color: #f0f0f0;
        }
        """

        self.setStyleSheet(stylesheet)
    
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = App()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
