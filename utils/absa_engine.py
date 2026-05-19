"""
ABSA Engine — Rule + VADER Hybrid Approach
Handles aspect detection, sentiment scoring, negation, highlighting
"""

import re
import math
from collections import defaultdict


class VADERLite:
    """
    Lightweight VADER-style lexicon scorer.
    Uses a curated food/product-domain lexicon.
    """
    LEXICON = {
        # Positive
        "amazing": 3.1, "wonderful": 3.0, "excellent": 3.2, "fantastic": 3.1,
        "great": 2.8, "good": 2.0, "delicious": 3.3, "tasty": 2.9, "yummy": 2.7,
        "fresh": 2.5, "love": 3.0, "loved": 3.0, "perfect": 3.2, "best": 3.0,
        "awesome": 3.1, "superb": 3.2, "outstanding": 3.3, "exceptional": 3.2,
        "pleased": 2.5, "happy": 2.6, "satisfied": 2.4, "recommend": 2.3,
        "beautiful": 2.8, "nice": 1.8, "clean": 1.9, "smooth": 2.0, "rich": 2.2,
        "crispy": 2.1, "crunchy": 2.0, "flavorful": 2.8, "aromatic": 2.3,
        "fast": 1.8, "quick": 1.7, "efficient": 2.0, "reliable": 2.1,
        "affordable": 2.2, "value": 1.9, "worth": 1.8, "genuine": 2.0,
        "healthy": 2.3, "natural": 2.0, "organic": 2.1, "pure": 2.0,
        "convenient": 1.9, "easy": 1.7, "simple": 1.5, "strong": 1.8,
        "impressive": 2.7, "delightful": 2.9, "enjoyable": 2.5, "pleasant": 2.3,

        # Negative
        "terrible": -3.2, "horrible": -3.3, "awful": -3.1, "disgusting": -3.4,
        "bad": -2.5, "poor": -2.3, "disappointing": -2.8, "disappointed": -2.7,
        "worst": -3.2, "nasty": -3.0, "gross": -2.9, "stale": -2.6,
        "bland": -2.2, "tasteless": -2.5, "soggy": -2.3, "mushy": -2.1,
        "rotten": -3.0, "spoiled": -3.1, "expired": -2.9, "mold": -3.2,
        "slow": -1.8, "late": -1.7, "delayed": -2.0, "broken": -2.5,
        "damaged": -2.3, "crushed": -2.4, "leaking": -2.6, "cheap": -1.8,
        "expensive": -1.5, "overpriced": -2.3, "waste": -2.5, "useless": -2.7,
        "hard": -1.5, "tough": -1.4, "chewy": -1.3, "dry": -1.8, "bitter": -2.0,
        "salty": -1.5, "burnt": -2.4, "oily": -1.7, "greasy": -2.0,
        "weak": -1.8, "thin": -1.3, "fake": -2.5, "artificial": -1.9,
        "unhealthy": -2.2, "processed": -1.7, "chemical": -2.0, "toxic": -3.0,
        "misleading": -2.5, "lie": -2.8, "scam": -3.0, "fraud": -3.1,
        "never": -1.5, "not": -1.0, "hate": -3.0, "dislike": -2.2,

        # Intensifiers
        "very": 0.7, "extremely": 1.2, "absolutely": 1.1, "totally": 0.9,
        "completely": 0.9, "really": 0.6, "so": 0.5, "quite": 0.4,
        "incredibly": 1.2, "surprisingly": 0.5, "highly": 0.8, "truly": 0.7,
    }

    NEGATORS = {"not", "no", "never", "don't", "doesn't", "didn't", "isn't",
                "aren't", "wasn't", "weren't", "won't", "wouldn't", "couldn't",
                "shouldn't", "cannot", "can't", "nothing", "neither", "nor",
                "without", "barely", "hardly", "scarcely", "rarely"}

    INTENSIFIERS = {"very", "extremely", "absolutely", "totally", "completely",
                    "really", "so", "quite", "incredibly", "highly", "truly", "super"}

    def score(self, text):
        tokens = re.findall(r"[a-z']+", text.lower())
        scores = []
        i = 0
        while i < len(tokens):
            word = tokens[i]
            if word in self.NEGATORS:
                i += 1
                # Negate next scored word
                if i < len(tokens) and tokens[i] in self.LEXICON:
                    scores.append(-self.LEXICON[tokens[i]] * 0.7)
                    i += 1
                continue
            if word in self.INTENSIFIERS and i + 1 < len(tokens):
                next_word = tokens[i + 1]
                if next_word in self.LEXICON:
                    scores.append(self.LEXICON[next_word] * 1.3)
                    i += 2
                    continue
            if word in self.LEXICON:
                scores.append(self.LEXICON[word])
            i += 1

        if not scores:
            return 0.0

        sum_s = sum(scores)
        alpha = 15
        compound = sum_s / math.sqrt(sum_s ** 2 + alpha)
        return round(compound, 4)


class ABSAEngine:
    """
    Main ABSA Engine: Aspect-Based Sentiment Analysis
    Uses rule-based aspect detection + VADER scoring
    """

    ASPECTS = {
        "taste": [
            "taste", "flavor", "flavour", "delicious", "yummy", "savory", "sweet",
            "bitter", "sour", "spicy", "bland", "rich", "fresh", "stale", "smell",
            "aroma", "scent", "fragrance", "odor", "palate", "zest"
        ],
        "quality": [
            "quality", "grade", "standard", "authentic", "genuine", "pure",
            "premium", "organic", "natural", "artificial", "processed", "ingredients",
            "fresh", "stale", "expired", "shelf life", "consistency", "material"
        ],
        "packaging": [
            "packaging", "package", "box", "bag", "container", "wrapper", "seal",
            "sealed", "lid", "jar", "bottle", "can", "tin", "damaged", "broken",
            "crushed", "dented", "torn", "wrapping", "label", "design"
        ],
        "price": [
            "price", "cost", "expensive", "cheap", "affordable", "worth", "value",
            "money", "dollars", "overpriced", "budget", "deal", "discount",
            "bargain", "fee", "rate", "pricing", "economical", "pricey"
        ],
        "shipping": [
            "shipping", "delivery", "arrived", "arrive", "shipped", "transit",
            "package", "days", "fast", "slow", "quick", "delayed", "late",
            "early", "tracking", "courier", "dispatch", "logistics", "fedex", "ups"
        ],
        "texture": [
            "texture", "consistency", "thick", "thin", "smooth", "rough", "soft",
            "hard", "crunchy", "crispy", "chewy", "mushy", "soggy", "dry", "moist",
            "creamy", "chunky", "gritty", "silky", "tender", "firm"
        ],
        "health": [
            "healthy", "unhealthy", "calories", "calorie", "sugar", "fat",
            "protein", "fiber", "sodium", "carbs", "gluten", "vegan", "vegetarian",
            "allergen", "nutrition", "vitamins", "ingredients", "diet", "organic",
            "chemical", "preservative", "additive", "gmo", "natural"
        ],
        "customer_service": [
            "service", "support", "customer", "refund", "return", "response",
            "seller", "vendor", "contact", "help", "assist", "resolve", "complaint",
            "experience", "staff", "team", "communication", "issue", "problem"
        ]
    }

    WINDOW_SIZE = 18  # words around aspect keyword to analyze

    def __init__(self):
        self.vader = VADERLite()

    def _split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def _detect_aspects_in_text(self, text):
        text_lower = text.lower()
        detected = {}
        for aspect, keywords in self.ASPECTS.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    if aspect not in detected:
                        detected[aspect] = []
                    detected[aspect].append(kw)
        return detected

    def _get_context_window(self, text, keyword, window=18):
        tokens = text.lower().split()
        kw_tokens = keyword.lower().split()
        
        for i in range(len(tokens)):
            if tokens[i:i+len(kw_tokens)] == kw_tokens:
                start = max(0, i - window)
                end = min(len(tokens), i + len(kw_tokens) + window)
                return " ".join(tokens[start:end])
        return text[:200]

    def _score_aspect_in_context(self, context):
        return self.vader.score(context)

    def analyze(self, text):
        sentences = self._split_sentences(text)
        
        aspect_scores_raw = defaultdict(list)
        aspect_spans = []
        
        for sentence in sentences:
            found = self._detect_aspects_in_text(sentence)
            for aspect, keywords in found.items():
                for kw in keywords:
                    context = self._get_context_window(sentence, kw)
                    score = self._score_aspect_in_context(context)
                    aspect_scores_raw[aspect].append(score)
                    
                    # Record span for highlighting
                    pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                    for match in pattern.finditer(text):
                        sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
                        aspect_spans.append({
                            "start": match.start(),
                            "end": match.end(),
                            "aspect": aspect,
                            "keyword": kw,
                            "sentiment": sentiment,
                            "score": score
                        })

        # Aggregate aspect scores
        aspect_scores = {}
        aspect_sentiments = {}
        for aspect, scores in aspect_scores_raw.items():
            avg = sum(scores) / len(scores)
            aspect_scores[aspect] = round(avg, 3)
            if avg > 0.05:
                aspect_sentiments[aspect] = "positive"
            elif avg < -0.05:
                aspect_sentiments[aspect] = "negative"
            else:
                aspect_sentiments[aspect] = "neutral"

        # Overall sentiment
        overall = self.vader.score(text)
        if overall > 0.05:
            overall_label = "positive"
        elif overall < -0.05:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        # Sentence breakdown
        sentence_breakdown = []
        for s in sentences:
            s_score = self.vader.score(s)
            sentence_breakdown.append({
                "text": s,
                "score": round(s_score, 3),
                "sentiment": "positive" if s_score > 0.05 else "negative" if s_score < -0.05 else "neutral"
            })

        return {
            "overall_sentiment": overall_label,
            "compound_score": round(overall, 4),
            "aspects": aspect_sentiments,
            "aspect_scores": aspect_scores,
            "aspect_spans": aspect_spans,
            "sentence_breakdown": sentence_breakdown
        }

    def highlight_text(self, text, spans):
        """Generate HTML with highlighted aspect mentions"""
        if not spans:
            return text

        # Sort spans by start, deduplicate
        seen = set()
        unique_spans = []
        for span in sorted(spans, key=lambda x: x['start']):
            key = (span['start'], span['end'])
            if key not in seen:
                seen.add(key)
                unique_spans.append(span)

        result = []
        last = 0
        for span in unique_spans:
            if span['start'] < last:
                continue
            result.append(text[last:span['start']])
            cls = f"highlight-{span['sentiment']}"
            title = f"{span['aspect']}: {span['sentiment']} ({span['score']:+.2f})"
            result.append(f'<span class="{cls}" title="{title}">{text[span["start"]:span["end"]]}</span>')
            last = span['end']
        result.append(text[last:])
        return "".join(result)

    def get_aspect_sentiment_summary(self, df):
        """Return per-aspect sentiment counts across dataset"""
        summary = {}
        for aspect in self.ASPECTS:
            pos = neg = neu = 0
            for _, row in df.iterrows():
                asp_sents = row.get('aspect_sentiments', {})
                if isinstance(asp_sents, dict) and aspect in asp_sents:
                    s = asp_sents[aspect]
                    if s == 'positive':
                        pos += 1
                    elif s == 'negative':
                        neg += 1
                    else:
                        neu += 1
            summary[aspect] = {'positive': pos, 'negative': neg, 'neutral': neu}
        return summary

    def get_aspect_frequency(self, df):
        """Return mention count per aspect"""
        freq = {}
        for aspect in self.ASPECTS:
            freq[aspect] = df['detected_aspects'].apply(
                lambda x: aspect in x if isinstance(x, list) else False
            ).sum()
        return freq
