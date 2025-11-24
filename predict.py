from tensorflow.keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import tensorflow.keras
import tensorflow.keras.backend as K
import os
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Bidirectional, LSTM, Reshape, Dropout
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading

prediction_model = None
num_to_char = None
vocab = None

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [50, 200])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = img.numpy()
    return img

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

def create_model():
    vocab_list = ["\u0425", "!", "\u043b", "N", "\u0414", "c", "\u041a", "'", "a", "5", "6", "s", "\u044b", "\u0417", "\u044e", "\u0445", ":", "\u041e", "\u0422", "\u0449", "\u0401", " ", "\u043a", "\u0441", "=", "+", "\u0432", "\u0426", "\u0444", "\u0447", "\u042b", "[", "\u0418", "B", "\u0433", "4", "\u0435", "\u0443", "7", "?", "\u044a", ")", "\u0442", "\u044c", "\u0427", "\u0424", "\u0411", "\u0437", "\u043c", "\u041c", "I", "O", "9", "\u0416", "\u042e", "}", "\u0429", "\u043d", "n", "3", ",", "\u0439", "\u044f", "]", "\u041f", "\u0438", "\u2116", "\u0421", "\"", "t", "V", "(", "\u043f", "\u0440", "e", "l", "r", "\u0448", "\u0431", "M", "/", "\u0415", "2", "\u042d", "\u0434", "\u0436", "_", "\u042f", "|", "\u0410", "0", "\u041b", "\u0420", "8", ";", "1", "-", "<", "\u0451", "\u0430", "z", "\u044d", "b", "\u0423", "\u0446", "\u0428", "\u0412", "\u043e", ">", ".", "\u041d", "\u0413", "T", "p", "*", "k", "y", "F", "A", "H", "u", "v", "g", "K", "f", "D", "d", "R", "L", "q", "\u042c", "Y", "X", "C", "i", "o", "S", "J", "G", "%", "w", "x", "U", "E", "j", "h", "m", "W", "P"]
    
    char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=vocab_list, mask_token=None)
    num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=char_to_num.get_vocabulary(), invert=True, mask_token=None)

    vgg = VGG16(include_top=False, input_shape=(200, 50, 3))

    conv1 = vgg.get_layer("block1_conv1")
    conv2 = vgg.get_layer("block1_conv2")
    pool1 = vgg.get_layer("block1_pool")

    conv3 = vgg.get_layer("block2_conv1")
    conv4 = vgg.get_layer("block2_conv2")
    pool2 = vgg.get_layer("block2_pool")

    img_input = Input(shape=(200, 50, 3), name="image_input", dtype="float32")
    lbl_input = Input(shape=(None,), dtype="float32")

    x = conv1(img_input)
    x = conv2(x)
    x = pool1(x)
    x = layers.BatchNormalization()(x)

    x = conv3(x)
    x = conv4(x)
    x = pool2(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(x)

    x = layers.BatchNormalization()(x)
    x = Reshape(((200 // 4), (50 // 4) * 64))(x)

    x = Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)

    x = Dense(151, activation="softmax", name="target_dense")(x)
    output = CTCLayer()(lbl_input, x)

    model = Model([img_input, lbl_input], output)
    model.compile(optimizer=tf.keras.optimizers.Adam())
    
    return model, num_to_char

def load_model_weights(model, weights_path="model.h5"):
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        return True
    return False

def decode_batch_predictions(pred, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :23
    ]

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res.replace("[UNK]", ""))
    return output_text

def recognize_images(image_paths, prediction_model, num_to_char, progress_callback=None):
    if not image_paths:
        return []
    
    images = []
    for path in image_paths:
        try:
            img = load_image(path)
            images.append(img)
        except Exception as e:
            print(f"Ошибка при загрузке изображения {path}: {e}")
            continue
    
    if not images:
        return []
    
    images = np.array(images)
    
    if progress_callback:
        progress_callback("Распознавание изображений...")
    
    prs = prediction_model.predict(images, verbose=0)
    pred_texts = decode_batch_predictions(prs, num_to_char)
    
    return pred_texts

class RecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание рукописного русского текста")
        self.root.geometry("900x700")
        self.root.configure(bg="#fafafa")
        
        self.colors = {
            'bg': '#fafafa',
            'primary': '#000000',
            'primary_hover': '#333333',
            'success': '#000000',
            'success_hover': '#333333',
            'danger': '#666666',
            'danger_hover': '#888888',
            'card_bg': '#ffffff',
            'text_primary': '#1a1a1a',
            'text_secondary': '#666666',
            'border': '#e5e5e5',
            'accent': '#000000'
        }
        
        self.image_paths = []
        self.prediction_model = None
        self.num_to_char = None
        self.model_loaded = False
        
        self.setup_ui()
        self.load_model_in_background()
    
    def create_button(self, parent, text, command, bg_color, hover_color, width=20):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10),
            bg=bg_color,
            fg="white",
            padx=20,
            pady=10,
            width=width,
            relief=tk.FLAT,
            cursor="hand2",
            bd=0,
            activebackground=hover_color,
            activeforeground="white"
        )
        
        def on_enter(e):
            btn.config(bg=hover_color)
        
        def on_leave(e):
            btn.config(bg=bg_color)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
    
    def create_card(self, parent, padx=20, pady=10):
        frame = tk.Frame(
            parent,
            bg=self.colors['card_bg'],
            relief=tk.FLAT,
            bd=1,
            highlightbackground=self.colors['border'],
            highlightthickness=0
        )
        return frame
    
    def setup_ui(self):
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        header_frame = self.create_card(main_container, pady=0)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text="Распознавание рукописного русского текста",
            font=("Segoe UI", 16),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            pady=24
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="Загрузите изображения и получите распознанный текст",
            font=("Segoe UI", 9),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        )
        subtitle_label.pack(pady=(0, 20))
        
        buttons_card = self.create_card(main_container)
        buttons_card.pack(fill=tk.X, pady=(0, 15))
        
        buttons_frame = tk.Frame(buttons_card, bg=self.colors['card_bg'])
        buttons_frame.pack(pady=16, padx=20)
        
        self.load_button = self.create_button(
            buttons_frame,
            "Загрузить изображения",
            self.load_images,
            self.colors['success'],
            self.colors['success_hover'],
            width=22
        )
        self.load_button.pack(side=tk.LEFT, padx=8)
        
        self.recognize_button = self.create_button(
            buttons_frame,
            "Распознать текст",
            self.recognize_images_threaded,
            self.colors['primary'],
            self.colors['primary_hover'],
            width=22
        )
        self.recognize_button.pack(side=tk.LEFT, padx=8)
        self.recognize_button.config(state=tk.DISABLED)
        
        clear_button = self.create_button(
            buttons_frame,
            "Очистить",
            self.clear_all,
            self.colors['danger'],
            self.colors['danger_hover'],
            width=14
        )
        clear_button.pack(side=tk.LEFT, padx=8)
        
        status_card = self.create_card(main_container)
        status_card.pack(fill=tk.X, pady=(0, 15))
        
        status_frame = tk.Frame(status_card, bg=self.colors['card_bg'])
        status_frame.pack(pady=12, padx=20, fill=tk.X)
        
        status_title = tk.Label(
            status_frame,
            text="Статус:",
            font=("Segoe UI", 9),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        )
        status_title.pack(side=tk.LEFT, padx=(0, 8))
        
        self.status_label = tk.Label(
            status_frame,
            text="Загрузка модели...",
            font=("Segoe UI", 9),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        )
        self.status_label.pack(side=tk.LEFT)
        
        self.progress = ttk.Progressbar(
            status_frame,
            mode='indeterminate',
            length=200
        )
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))
        
        images_card = self.create_card(main_container)
        images_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        images_header = tk.Frame(images_card, bg=self.colors['card_bg'])
        images_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        images_title = tk.Label(
            images_header,
            text="Загруженные изображения",
            font=("Segoe UI", 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        )
        images_title.pack(side=tk.LEFT)
        
        count_label_frame = tk.Frame(images_header, bg=self.colors['card_bg'])
        count_label_frame.pack(side=tk.RIGHT)
        
        self.count_label = tk.Label(
            count_label_frame,
            text="0 файлов",
            font=("Segoe UI", 9),
            bg=self.colors['card_bg'],
            fg=self.colors['text_secondary']
        )
        self.count_label.pack()
        
        listbox_frame = tk.Frame(images_card, bg=self.colors['card_bg'])
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.images_listbox = tk.Listbox(
            listbox_frame,
            font=("Segoe UI", 9),
            bg="#ffffff",
            fg=self.colors['text_primary'],
            selectbackground=self.colors['primary'],
            selectforeground="white",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            yscrollcommand=scrollbar.set,
            height=4
        )
        self.images_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.images_listbox.yview)
        
        results_card = self.create_card(main_container)
        results_card.pack(fill=tk.BOTH, expand=True)
        
        results_header = tk.Frame(results_card, bg=self.colors['card_bg'])
        results_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        results_title = tk.Label(
            results_header,
            text="Результаты распознавания",
            font=("Segoe UI", 10),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary']
        )
        results_title.pack(side=tk.LEFT)
        
        text_frame = tk.Frame(results_card, bg=self.colors['card_bg'])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        self.results_text = scrolledtext.ScrolledText(
            text_frame,
            font=("Segoe UI", 10),
            bg="#ffffff",
            fg=self.colors['text_primary'],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            wrap=tk.WORD,
            padx=16,
            pady=16
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        self.results_text.insert(tk.END, "Здесь будут отображаться результаты распознавания текста...\n\n")
        self.results_text.config(state=tk.DISABLED)
    
    def load_model_in_background(self):
        def load():
            try:
                self.root.after(0, lambda: self.status_label.config(text="Создание модели...", fg=self.colors['text_secondary']))
                self.root.after(0, lambda: self.progress.start())
                
                model, num_to_char = create_model()
                
                self.root.after(0, lambda: self.status_label.config(text="Загрузка весов модели...", fg=self.colors['text_secondary']))
                
                if load_model_weights(model, "model.h5"):
                    self.prediction_model = tf.keras.models.Model(
                        model.get_layer(name="image_input").input, 
                        model.get_layer(name="target_dense").output
                    )
                    self.num_to_char = num_to_char
                    self.model_loaded = True
                    
                    self.root.after(0, lambda: self.progress.stop())
                    self.root.after(0, lambda: self.status_label.config(text="Модель загружена успешно", fg=self.colors['text_primary']))
                    
                    if self.image_paths:
                        self.root.after(0, lambda: self.recognize_button.config(state=tk.NORMAL))
                else:
                    self.root.after(0, lambda: self.progress.stop())
                    self.root.after(0, lambda: self.status_label.config(text="Ошибка: файл model.h5 не найден", fg=self.colors['text_primary']))
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", "Файл model.h5 не найден в текущей директории!"))
            except Exception as e:
                self.root.after(0, lambda: self.progress.stop())
                error_msg = f"Ошибка загрузки модели: {str(e)}"
                self.root.after(0, lambda: self.status_label.config(text=error_msg, fg=self.colors['text_primary']))
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{str(e)}"))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def load_images(self):
        filetypes = [
            ("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("PNG файлы", "*.png"),
            ("JPEG файлы", "*.jpg *.jpeg"),
            ("Все файлы", "*.*")
        ]
        
        paths = filedialog.askopenfilenames(
            title="Выберите изображения для распознавания",
            filetypes=filetypes
        )
        
        if paths:
            self.image_paths = list(paths)
            self.images_listbox.delete(0, tk.END)
            for path in self.image_paths:
                filename = os.path.basename(path)
                self.images_listbox.insert(tk.END, filename)
            
            self.count_label.config(text=f"{len(self.image_paths)} файлов")
            
            if self.model_loaded:
                self.recognize_button.config(state=tk.NORMAL)
            else:
                self.status_label.config(text="Ожидание загрузки модели...", fg=self.colors['text_secondary'])
    
    def recognize_images_threaded(self):
        if not self.image_paths:
            messagebox.showwarning("Предупреждение", "Пожалуйста, сначала загрузите изображения!")
            return
        
        if not self.model_loaded:
            messagebox.showwarning("Предупреждение", "Модель еще не загружена. Пожалуйста, подождите.")
            return
        
        def recognize():
            try:
                self.root.after(0, lambda: self.recognize_button.config(state=tk.DISABLED))
                self.root.after(0, lambda: self.status_label.config(text="Распознавание...", fg=self.colors['text_secondary']))
                self.root.after(0, lambda: self.progress.start())
                
                def progress_callback(message):
                    self.root.after(0, lambda: self.status_label.config(text=message, fg=self.colors['text_secondary']))
                
                results = recognize_images(self.image_paths, self.prediction_model, 
                                          self.num_to_char, progress_callback)
                
                self.root.after(0, lambda: self.display_results(results))
                
            except Exception as e:
                error_msg = f"Ошибка при распознавании: {str(e)}"
                self.root.after(0, lambda: self.progress.stop())
                self.root.after(0, lambda: self.status_label.config(text=error_msg, fg=self.colors['text_primary']))
                self.root.after(0, lambda: messagebox.showerror("Ошибка", error_msg))
                self.root.after(0, lambda: self.recognize_button.config(state=tk.NORMAL))
        
        thread = threading.Thread(target=recognize, daemon=True)
        thread.start()
    
    def display_results(self, results):
        self.progress.stop()
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        if not results:
            self.results_text.insert(tk.END, "Не удалось распознать текст на изображениях.\n")
            self.status_label.config(text="Распознавание завершено с ошибками", fg=self.colors['text_primary'])
        else:
            for i, (path, text) in enumerate(zip(self.image_paths, results), 1):
                filename = os.path.basename(path)
                self.results_text.insert(tk.END, f"Изображение {i}: {filename}\n", "header")
                self.results_text.insert(tk.END, f"Распознанный текст:\n", "label")
                self.results_text.insert(tk.END, f"{text}\n\n", "text")
                self.results_text.insert(tk.END, "\n", "separator")
            
            self.status_label.config(text=f"Распознавание завершено. Обработано изображений: {len(results)}", fg=self.colors['text_primary'])
        
        self.results_text.tag_config("header", font=("Segoe UI", 10), foreground=self.colors['text_primary'])
        self.results_text.tag_config("label", font=("Segoe UI", 9), foreground=self.colors['text_secondary'])
        self.results_text.tag_config("text", font=("Segoe UI", 10), foreground=self.colors['text_primary'])
        self.results_text.tag_config("separator", foreground=self.colors['border'])
        
        self.results_text.config(state=tk.DISABLED)
        self.recognize_button.config(state=tk.NORMAL)
    
    def clear_all(self):
        self.image_paths = []
        self.images_listbox.delete(0, tk.END)
        self.count_label.config(text="0 файлов")
        
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Здесь будут отображаться результаты распознавания текста...\n\n")
        self.results_text.config(state=tk.DISABLED)
        
        if self.model_loaded:
            self.status_label.config(text="Модель готова к работе", fg=self.colors['text_primary'])
            self.recognize_button.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = RecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()