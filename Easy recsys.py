import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.preprocessing import LabelEncoder
import dask.dataframe as dd

def recommend(data_url):
  
  df1 = dd.read_csv(data_url)
  df = df1.head(10000)


  
  col = df.select_dtypes(['object']).columns
  le=LabelEncoder()
  for i in col:
      df[i]=le.fit_transform(df[i])
  df.head()
  normalized_df = normalize(df)

  
  svd = TruncatedSVD(n_components=2)
  transformed_data = svd.fit_transform(normalized_df)


  def similarity(vector1, vector2):
    return np.dot(vector1, vector2)


  def get_recommendations(item_index):
    item_vector = transformed_data[item_index]
    similarities = [similarity(item_vector, other_vector) for other_vector in transformed_data]
    most_similar_items = np.argsort(similarities)[::-1][1:6] 
    return most_similar_items

  item_index = st.number_input("Введите индекс элемента для рекомендаций", min_value=0, max_value=len(df)-1)


  if st.button("Выдать рекомендации"):
    recommended_items = get_recommendations(item_index)
    st.write("Данные выбранного элемента:")
    st.write(df.iloc[item_index])
    st.write("Рекомендованные элементы:", recommended_items)
    st.write(df.iloc[recommended_items])


st.title("Система рекомендаций на основе SVD")
data_url = st.text_input("Вставьте ссылку на данные (CSV)")

if data_url:
  recommend(data_url)
