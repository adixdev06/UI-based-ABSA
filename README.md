# 🔬 Aspect-Based Sentiment Analysis (ABSA)
### Amazon Fine Food Reviews · Python + Streamlit · Dark UI

---

## 📦 Dataset

**Amazon Fine Food Reviews** — Kaggle  
🔗 https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

| Field | Info |
|---|---|
| Size | 568,454 reviews |
| Period | Oct 1999 – Oct 2012 |
| Source | Amazon.com |
| Columns | Id, ProductId, UserId, Score, Summary, Text, etc. |

The app ships with a **built-in synthetic dataset** so it works immediately.  
To use the real Kaggle data, follow the setup steps below.

---

## 🚀 Quick Start

```bash
# 1. Clone / unzip the project
cd absa_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📁 Project Structure

```
absa_project/
├── app.py                    ← Main Streamlit app (5 pages)
├── requirements.txt
├── README.md
├── data/
│   └── Reviews.csv           ← (Place Kaggle CSV here)
└── utils/
    ├── __init__.py
    ├── absa_engine.py        ← Core ABSA logic (aspect detection + scoring)
    ├── data_loader.py        ← Dataset loading + preprocessing pipeline
    └── visualizer.py        ← Plotly chart helpers
```

---

## 🔌 Using Real Kaggle Data

1. Go to https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
2. Download `Reviews.csv`
3. Place it in the `data/` folder
4. Open `utils/data_loader.py` and set:
   ```python
   USE_REAL_DATA = True
   SAMPLE_SIZE = 1000   # or more if your machine has RAM
   ```
5. Delete `data/processed_reviews.pkl` if it exists (clears cache)
6. Restart the app: `streamlit run app.py`

---

## 🧠 How ABSA Works

### Pipeline

```
Input Review
    ↓
Text Preprocessing (tokenize, sentence split)
    ↓
Aspect Detection (keyword matching with 8 aspect categories)
    ↓
Context Window Extraction (±18 words around each aspect keyword)
    ↓
VADER Sentiment Scoring (per context window)
    ↓
Negation Handling (flips polarity for "not", "never", etc.)
    ↓
Aspect → Sentiment Mapping + Compound Score
```

### Aspect Categories

| Aspect | Example Keywords |
|---|---|
| **Taste** | flavor, delicious, bland, aroma, zest |
| **Quality** | organic, fresh, expired, authentic, premium |
| **Packaging** | box, sealed, damaged, crushed, jar |
| **Price** | expensive, affordable, value, overpriced |
| **Shipping** | delivery, arrived, fast, delayed, tracking |
| **Texture** | crunchy, smooth, soggy, chewy, creamy |
| **Health** | calories, organic, preservative, gluten, vegan |
| **Customer Service** | refund, support, seller, response, complaint |

### Sentiment Thresholds

| Compound Score | Label |
|---|---|
| > +0.05 | Positive |
| < −0.05 | Negative |
| −0.05 to +0.05 | Neutral |

---

## 🖥️ App Pages

| Page | What You Get |
|---|---|
| 🏠 Dashboard | Overview stats, stacked bar chart, pie chart, top aspects |
| 📝 Analyze Review | Paste any review → get aspect-level results, radar chart, highlights |
| 📊 Dataset Explorer | Filter & browse the full dataset, score distribution |
| 🏷️ Aspect Deep Dive | Select one aspect → see sentiment breakdown + sample reviews |
| 📈 Model Insights | Pipeline diagram, keyword list, methodology notes |

---

## 🔧 Upgrading to Transformer Model

To replace the lexicon-based VADER scorer with a BERT model:

```python
# In utils/absa_engine.py, replace VADERLite with:
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model="yangheng/deberta-v3-base-absa-v1.1"   # ABSA-specific model
)
```

Recommended models on HuggingFace:
- `yangheng/deberta-v3-base-absa-v1.1`
- `nickmuchi/deberta-v3-base-finetuned-finance-text-classification`
- `lxyuan/distilbert-base-multilingual-cased-sentiments-student`

---

## 📚 References

- Pontiki et al., *SemEval-2014 Task 4: Aspect Based Sentiment Analysis*
- Hutto & Gilbert, *VADER: A Parsimonious Rule-based Model for Sentiment Analysis* (AAAI 2014)
- Liu, *Sentiment Analysis and Opinion Mining*, Morgan & Claypool, 2012

---

## 📄 License

MIT — free to use, modify, and distribute.
