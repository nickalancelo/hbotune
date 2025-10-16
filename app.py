# -*- coding: utf-8 -*-
# app.py — HBOTune (RF tuned model) • Form GET/POST + opsiyonel JSON API
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# ---- Model ayarları ----
MODEL_PATH = "rf_model_tuned.pkl"  # tune betiğinin kaydettiği model
FEATURES = [
    "Yaş",
    "Cinsiyet",
    "Tedavi Gecikmesi (gün)",
    "HBOT Seans Sayısı",
    "Ek Hastalık",
    "Sigara Kullanımı",
    "Sistemik Steroid Kullanımı",
    "Intratimpanik Steroid Kullanımı",
    "Başlangıç PTA (dB)",
]

def to01(v):
    """Evet/Hayır, Var/Yok, Erkek/Kadın gibi girdileri 0/1'e çevirir."""
    s = str(v).strip().lower().replace("ı", "i")
    s = s.translate(str.maketrans("çğıöşü", "cgiosu"))
    if s in ["1","evet","var","yes","true","pozitif","positive","erkek","e","male","m"]:
        return 1
    if s in ["0","hayir","yok","no","false","negatif","negative","kadin","k","female","f"]:
        return 0
    try:
        return 1 if float(str(v).replace(",", ".")) > 0 else 0
    except:
        return None

def preprocess_one(input_dict):
    """Form/API girdisini eğitim formatına çevirir (tek satırlık DataFrame)."""
    df = pd.DataFrame([input_dict])

    # kategorikleri 0/1
    for c in ["Cinsiyet","Ek Hastalık","Sigara Kullanımı",
              "Sistemik Steroid Kullanımı","Intratimpanik Steroid Kullanımı"]:
        df[c] = df.get(c).apply(to01) if c in df else None

    # sayısallar
    for c in ["Yaş","Tedavi Gecikmesi (gün)","HBOT Seans Sayısı","Başlangıç PTA (dB)"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce") if c in df else None

    # eğitimdeki sıra
    df = df[FEATURES]

    # eksik kontrol
    if df.isna().any().any():
        eksik = [c for c in FEATURES if df[c].isna().any()]
        raise ValueError("Eksik/yanlış alanlar: " + ", ".join(eksik))
    return df

# modeli yükle
model = joblib.load(MODEL_PATH)

def predict_gain(input_dict) -> float:
    X = preprocess_one(input_dict)
    return float(model.predict(X)[0])

# ---- Flask app ----
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            y_hat = round(predict_gain(form), 2)
            return render_template("index.html", prediction=y_hat, form=form)
        except Exception as e:
            return render_template("index.html", error=str(e), form=form), 400
    # GET
    return render_template("index.html")

# İsteğe bağlı: JSON API (Ajax/entegrasyonlar için)
@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json(force=True)
        y_hat = round(predict_gain(data), 2)
        return jsonify({"tahmin_dB": y_hat})
    except Exception as e:
        return jsonify({"hata": str(e)}), 400

if __name__ == "__main__":
    # Çalıştır: python app.py  → http://localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=False)
