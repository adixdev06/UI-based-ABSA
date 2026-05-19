"""
DataLoader — Generates a realistic Amazon Fine Food Reviews dataset.

REAL DATASET (recommended):
    Download from Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
    Place Reviews.csv in the data/ folder, then set USE_REAL_DATA = True below.

This module also generates a synthetic sample (500 reviews) so the app works
out-of-the-box without the download.
"""

import pandas as pd
import random
import re
import sys
import os
from collections import defaultdict

# ── Try to import from parent package ──
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.absa_engine import ABSAEngine

USE_REAL_DATA = False      # Set True after placing Reviews.csv in data/
REAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Reviews.csv")
SAMPLE_SIZE = 500          # rows to use from real data (or synthetic rows to generate)
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_reviews.pkl")


# ── Synthetic review templates ──────────────────────────────────
POS_PHRASES = [
    "The taste is absolutely amazing and really rich.",
    "Flavor is fantastic — better than expected.",
    "Incredibly fresh and the aroma is wonderful.",
    "Great quality product, completely satisfied.",
    "The texture is perfectly smooth and creamy.",
    "Packaging was excellent, arrived in perfect condition.",
    "Super fast delivery, came well ahead of schedule.",
    "Very affordable for the quality you get.",
    "Healthy ingredients, clean label, no artificial stuff.",
    "Customer service was outstanding — helped right away.",
    "Would highly recommend to anyone who loves good food.",
    "Best product I have tried in this category so far.",
]
NEG_PHRASES = [
    "The taste was absolutely terrible and very bland.",
    "Flavor was disappointing — nothing like described.",
    "The smell was off-putting and quite disgusting.",
    "Poor quality, definitely not worth the price.",
    "Texture was soggy and mushy — completely wrong.",
    "Packaging was crushed and damaged on arrival.",
    "Shipping was extremely slow, took over two weeks.",
    "Way too expensive for such a small portion size.",
    "Full of artificial preservatives and chemicals.",
    "Customer service never responded to my complaint.",
    "Do not recommend this product to anyone.",
    "Worst purchase I have made this year.",
]
NEU_PHRASES = [
    "The taste is okay, nothing special about it.",
    "Flavor is decent, but not outstanding either.",
    "Quality seems average for the price range.",
    "Packaging is standard, no complaints or praise.",
    "Delivery time was about what I expected.",
    "Price is in line with similar products out there.",
    "Ingredients look normal, nothing alarming here.",
    "Customer service was polite but not super helpful.",
]
MIXED_PHRASES = [
    ("The taste is amazing but the packaging was terrible.", 4),
    ("Flavor is great though delivery was really slow.", 3),
    ("Excellent quality ingredients, but way overpriced.", 3),
    ("Texture is perfect but the smell is a bit off.", 3),
    ("Fast shipping and great service, but taste was bland.", 3),
    ("Good value for money though quality could be better.", 3),
    ("Beautiful packaging but the product itself was disappointing.", 2),
    ("Taste is decent but the texture was too soft and mushy.", 3),
]

PRODUCT_NAMES = [
    "Himalayan Pink Salt", "Organic Coffee Beans", "Almond Flour",
    "Dark Chocolate Bar", "Gummy Bears", "Protein Powder", "Olive Oil",
    "Matcha Green Tea", "Beef Jerky", "Coconut Oil", "Granola Bars",
    "Hot Sauce", "Peanut Butter", "Honey", "Trail Mix"
]


class DataLoader:

    def __init__(self):
        self.engine = ABSAEngine()

    def load_dataset(self):
        """Load real or synthetic dataset, with caching."""
        if os.path.exists(CACHE_PATH):
            return pd.read_pickle(CACHE_PATH)

        if USE_REAL_DATA and os.path.exists(REAL_DATA_PATH):
            df = self._load_real(REAL_DATA_PATH)
        else:
            df = self._generate_synthetic(SAMPLE_SIZE)

        df = self._preprocess(df)
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_pickle(CACHE_PATH)
        return df

    # ── Real data ────────────────────────────────────────────────
    def _load_real(self, path):
        print(f"Loading real dataset from {path}...")
        df = pd.read_csv(path, nrows=SAMPLE_SIZE)
        df = df[['Score', 'Summary', 'Text']].dropna()
        df['Text'] = df['Text'].astype(str)
        return df

    # ── Synthetic data ───────────────────────────────────────────
    def _generate_synthetic(self, n):
        records = []
        random.seed(42)

        for i in range(n):
            r = random.random()
            if r < 0.45:          # positive
                score = random.choice([4, 5])
                parts = random.sample(POS_PHRASES, k=random.randint(2, 4))
                text = " ".join(parts)
            elif r < 0.70:        # negative
                score = random.choice([1, 2])
                parts = random.sample(NEG_PHRASES, k=random.randint(2, 4))
                text = " ".join(parts)
            elif r < 0.80:        # neutral
                score = 3
                parts = random.sample(NEU_PHRASES, k=random.randint(2, 3))
                text = " ".join(parts)
            else:                  # mixed
                template, score = random.choice(MIXED_PHRASES)
                extra = random.choice(POS_PHRASES + NEG_PHRASES)
                text = template + " " + extra

            product = random.choice(PRODUCT_NAMES)
            text = f"I bought {product}. {text}"

            records.append({"Score": score, "Text": text})

        return pd.DataFrame(records)

    # ── Preprocessing ─────────────────────────────────────────────
    def _preprocess(self, df):
        """Run ABSA on every row and attach derived columns."""
        results = []
        for _, row in df.iterrows():
            text = str(row['Text'])[:600]   # cap length for speed
            res = self.engine.analyze(text)
            results.append({
                "compound_score":    res['compound_score'],
                "sentiment_label":   res['overall_sentiment'],
                "detected_aspects":  list(res['aspects'].keys()),
                "aspect_sentiments": res['aspects'],
                "aspect_count":      len(res['aspects']),
            })

        derived = pd.DataFrame(results)
        df = df.reset_index(drop=True)
        return pd.concat([df, derived], axis=1)
