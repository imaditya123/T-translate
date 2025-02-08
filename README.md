# T-Translate: Transformer Model from Scratch ⚡

T-Translate is a **Transformer-based text generation model** built **from scratch** using PyTorch. It follows the original **Attention Is All You Need** paper, implementing a **custom Transformer architecture** for text-to-text tasks such as machine translation or text completion.

---

## **✨ Features**  
✅ Custom-built Transformer model (Encoder-Decoder) from scratch  
✅ Uses PyTorch for efficient training and inference  
✅ Supports text-to-text tasks like translation
✅ Configurable hyperparameters via `config.py`  
✅ Includes training, evaluation, and inference scripts  

---

## **📂 Project Structure**  
```
T-Translate/
│── models/             # Saved trained models
│── src/                # Source code
│   ├── config.py       # Hyperparameter settings
│   ├── model.py        # Custom Transformer model
│   ├── preprocess.py   # Data preprocessing utilities
│   ├── train.py        # Training script
│   ├── evaluate.py     # Model evaluation
│   ├── translate.py    # Inference script
│── notebooks/          # Jupyter notebooks for experiments
│── tests/              # Unit tests for validation
│── requirements.txt    # Dependencies
│── README.md           # Documentation
│── .gitignore          # Ignore unnecessary files
```

---

## **🔧 Setup & Installation**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/imaditya123/T-translate.git
cd T-Translate
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Train the Model**  
```bash
python src/train.py
```

### **4️⃣ Run Inference**  
```bash
python src/translate.py --text "Hello, world!"
```

---

## **⚙️ Configuration (Modify `config.py`)**  
You can adjust **hyperparameters** like:  
```python
MODEL_DIM = 512         # Transformer hidden size
NUM_LAYERS = 6          # Number of encoder & decoder layers
NUM_HEADS = 8           # Multi-head attention heads
DROPOUT = 0.1           # Dropout rate
BATCH_SIZE = 32         # Training batch size
LEARNING_RATE = 3e-4    # Optimizer learning rate
EPOCHS = 10             # Number of training epochs
```

---

## **🚀 Future Enhancements**  
🔹 Add support for sequence classification tasks  
🔹 Optimize for large-scale datasets  
🔹 Deploy as an API  

---

## **📜 References**  
- Vaswani et al., **Attention Is All You Need** (2017)  
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html  

📢 **Contributions & feedback are welcome!** Open an issue or create a pull request. 🚀

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/imaditya123/T-translate?tab=MIT-1-ov-file) file for details.

---