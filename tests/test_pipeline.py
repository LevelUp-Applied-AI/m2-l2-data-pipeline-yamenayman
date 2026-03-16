"""
Lab 2 — Learner Test File

Write your own pytest tests here. You must implement at least 3 test functions:
  - test_load_data_returns_dataframe
  - test_clean_data_no_nulls
  - test_add_features_creates_revenue

The autograder will run your tests as part of the CI check.
"""

import pandas as pd
import numpy as np
import pytest
from pipeline import load_data, clean_data, add_features

# تحديد مسار الملف لاستخدامه في الاختبارات
TEST_DATA_PATH = 'data/sales_records.csv'

# ─── Test 1 ───────────────────────────────────────────────────────────────────

def test_load_data_returns_dataframe():
    """load_data should return a DataFrame with expected columns and rows."""
    df = load_data(TEST_DATA_PATH)
    
    # التحقق من أن النتيجة هي DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # التحقق من أن عدد الصفوف أكبر من صفر
    assert len(df) > 0
    
    # التحقق من وجود جميع الأعمدة المتوقعة
    expected_columns = ['date', 'store_id', 'product_category', 'quantity', 'unit_price', 'payment_method']
    for col in expected_columns:
        assert col in df.columns

# ─── Test 2 ───────────────────────────────────────────────────────────────────

def test_clean_data_no_nulls():
    """After clean_data, quantity and unit_price should have no NaN values."""
    # تحميل وتنظيف البيانات
    df = load_data(TEST_DATA_PATH)
    cleaned_df = clean_data(df)
    
    # التحقق من عدم وجود قيم مفقودة في الكمية والسعر
    assert cleaned_df['quantity'].isna().sum() == 0
    assert cleaned_df['unit_price'].isna().sum() == 0

# ─── Test 3 ───────────────────────────────────────────────────────────────────

def test_add_features_creates_revenue():
    """add_features should add a 'revenue' column equal to quantity * unit_price."""
    # تحميل وتنظيف وإضافة الميزات
    df = load_data(TEST_DATA_PATH)
    cleaned_df = clean_data(df)
    enriched_df = add_features(cleaned_df)
    
    # التحقق من وجود عمود الإيرادات
    assert 'revenue' in enriched_df.columns
    
    # حساب الإيرادات المتوقعة
    expected_revenue = enriched_df['quantity'] * enriched_df['unit_price']
    
    # التحقق من تطابق القيم باستخدام أداة pandas للمقارنة الدقيقة للأرقام العشرية
    pd.testing.assert_series_equal(enriched_df['revenue'], expected_revenue, check_names=False)