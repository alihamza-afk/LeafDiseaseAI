# 🌿 Leaf Disease AI

A Streamlit web app that identifies plant leaf diseases from a single photo using a fine-tuned **InceptionV3** deep learning model trained on 38 disease classes across 14 crop species.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://leafdiseaseai.streamlit.app)

---

## ✨ Features

- Upload any leaf photo (JPG / PNG) and get an instant prediction
- Displays the top predicted disease with a model confidence score
- Shows two alternative predictions for transparency
- Warns you when the model is uncertain (score < 50%)
- Runs locally or deploys in one click to Streamlit Community Cloud

---

## 🌱 Supported Plants & Diseases (38 classes)

| Plant | Conditions detected |
|-------|-------------------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| Orange | Huanglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Bell Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🧠 Model Architecture

| Property | Value |
|----------|-------|
| Base model | InceptionV3 (ImageNet pre-trained) |
| Input size | 224 × 224 × 3 |
| Fine-tuning strategy | Feature extraction (base frozen) + custom classification head |
| Output classes | 38 |
| Total parameters | ~21.9 M (77 K trainable) |
| Augmentation | Random horizontal flip, random zoom ±20% |
| Loss | Categorical cross-entropy |
| Optimizer | Adam (lr = 1e-4) |
| Training dataset | 70,295 images across 38 classes |
| Validation dataset | 17,572 images |

The model was trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle, which is an augmented version of the original PlantVillage dataset.

---

## 🚀 Run Locally

**Requirements:** Python 3.11

```bash
# Clone the repo
git clone https://github.com/alihamza-afk/LeafDiseaseAI.git
cd LeafDiseaseAI

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note for Apple Silicon (M1/M2/M3) users:** The `requirements.txt` already includes `tensorflow-macos` and `tensorflow-metal` for GPU-accelerated inference on Apple Silicon.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Fork or push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**.
3. Select this repository, branch `main`, and `app.py` as the entry point.
4. Under **Advanced settings**, set the Python version to **3.11** — TensorFlow 2.15 does not support Python 3.12.
5. Click **Deploy**. The trained model and class labels are bundled in the repo, so no extra setup is needed.

---

## 📁 Project Structure

```
LeafDiseaseAI/
├── app.py                      # Streamlit application
├── leaf_disease_final.keras    # Trained model (used by the app)
├── leaf_disease_model.keras    # Intermediate checkpoint
├── class_names.json            # Ordered list of 38 class labels
├── requirements.txt            # Python dependencies
├── training_notebook.ipynb     # Model training notebook (InceptionV3 fine-tuning)
└── README.md
```

---

## ⚠️ Disclaimer

This tool is intended for educational and research purposes only. Model scores are raw softmax outputs and are **not** calibrated probabilities. This app is **not** a substitute for professional agricultural or plant pathology advice. Always consult a qualified expert for actual crop disease diagnosis and treatment decisions.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The training dataset ([New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)) is subject to its own license on Kaggle.
