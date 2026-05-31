"""
features.py - shared feature extractor for train.py and app.py.

Keeping this in a shared module ensures pickle can find the class when loading model.pkl.
"""

import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class CombinedFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Combines TF-IDF with 15 Indian fraud hand-crafted features.

    Input:  DataFrame with at least full_text and company columns.
    Output: sparse matrix (tfidf features || hand features)
    """

    PATTERNS = {
        # Fee / payment signals
        "fee_language":         r"registration fee|processing fee|consultancy fee|security deposit|refundable deposit|training fee|joining fee",
        "upfront_payment":      r"pay.{0,20}(first|advance|before joining|before training)|deposit.{0,20}refund",

        # Unrealistic compensation
        "unrealistic_salary":   r"₹\s*[5-9]\d{3,}[\s/]*(day|daily)|earn.{0,15}₹\s*[5-9]\d{3,}|daily earning|per day earning",
        "guaranteed_job":       r"100%\s*(job\s*)?(guarantee|guaranteed)|placement guarantee|guaranteed placement|assured job",

        # Contact red flags
        "whatsapp_only":        r"whatsapp\s*(only|me|us|number|contact)|contact.{0,20}whatsapp",
        "gmail_contact":        r"[\w.\-]+@gmail\.com|[\w.\-]+@yahoo\.com|[\w.\-]+@hotmail\.com",
        "personal_phone":       r"(call|contact|whatsapp).{0,20}\+?91[-\s]?[6-9]\d{9}",

        # Document / identity harvesting
        "aadhaar_request":      r"aadhaar|aadhar|uid number",
        "pan_request":          r"\bpan\s*(card|number|details)\b",
        "bank_details_early":   r"bank\s*(account|details|statement).{0,40}(apply|before|joining|interview|send)",

        # Fake MNC / impersonation
        "mnc_impersonation":    r"\b(tcs|infosys|wipro|accenture|amazon|flipkart|hcl|tech mahindra)\b.{0,60}(hiring|recruit|opening|vacancy)",
        "background_check_fee": r"background\s*(check|verification).{0,30}(fee|charge|pay|₹)",

        # Overseas scam signals
        "overseas_visa_fee":    r"(dubai|singapore|canada|uk|abroad|overseas).{0,60}(visa|processing).{0,30}(fee|charge|₹)",

        # WFH task scam
        "simple_task_scam":     r"(like|share|follow|click).{0,20}(earn|money|₹|income)|work from (home|mobile).{0,30}(₹|earn|income).{0,20}(day|daily|hour)",

        # Structural vagueness
        "vague_company":        r"^.{0,30}$",
    }

    def __init__(self, max_features=30000):
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
        )

    def fit(self, X, y=None):
        self.tfidf.fit(X["full_text"].fillna("").tolist())
        return self

    def _hand_features(self, X):
        texts = X["full_text"].fillna("").str.lower().tolist()
        companies = X["company"].fillna("").str.lower().tolist()
        rows = []
        for text, company in zip(texts, companies):
            row = []
            for name, pat in self.PATTERNS.items():
                if name == "vague_company":
                    row.append(int(bool(re.match(pat, company.strip()))))
                else:
                    row.append(int(bool(re.search(pat, text, re.IGNORECASE))))
            total_flags = sum(row)
            flag_density = total_flags / max(len(text.split()), 1)
            row += [total_flags, flag_density]
            rows.append(row)
        return sp.csr_matrix(np.array(rows, dtype=float))

    def transform(self, X):
        tfidf_matrix = self.tfidf.transform(X["full_text"].fillna("").tolist())
        hand_matrix = self._hand_features(X)
        return sp.hstack([tfidf_matrix, hand_matrix])

    def get_feature_names_out(self):
        tfidf_names = list(self.tfidf.get_feature_names_out())
        hand_names = [f"indian_{k}" for k in self.PATTERNS]
        hand_names += ["indian_total_flags", "indian_flag_density"]
        return tfidf_names + hand_names