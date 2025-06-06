import pandas as pd
import ast
import html
import re

def parse_list_string(s):
    if isinstance(s, str) and s.startswith('[') and s.endswith(']'):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    return []

def clean_review_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

def load_and_prepare_data(use_sample=True):
    nrows_to_load = 200000 if use_sample else None

    print(f"Carregando dados... Modo de amostragem: {use_sample}, Linhas: {nrows_to_load or 'Todas'}")

    ratings_df = pd.read_csv('data/Books_rating.csv', nrows=nrows_to_load)
    metadata_df = pd.read_csv('data/books_data.csv')
    
    ratings_df = ratings_df.drop_duplicates(subset=['User_id', 'Title', 'text', 'score', 'time'], keep='first')
    ratings_df = ratings_df.dropna(subset=['Title', 'User_id'])
    ratings_df['summary'] = ratings_df['summary'].fillna('')
    ratings_df['text'] = ratings_df['text'].fillna('')

    metadata_df = metadata_df.dropna(subset=['Title', 'authors'])
    metadata_df['publisher'] = metadata_df['publisher'].fillna('Editora desconhecida')
    metadata_df['description'] = metadata_df['description'].fillna('Descrição não disponível')
    
    metadata_df['authors'] = metadata_df['authors'].apply(parse_list_string)
    metadata_df['categories'] = metadata_df['categories'].apply(parse_list_string)
    
    metadata_df.loc[metadata_df['authors'].apply(len) == 0, 'authors'] = pd.Series([['Autor Desconhecido']] * len(metadata_df))
    metadata_df.loc[metadata_df['categories'].apply(len) == 0, 'categories'] = pd.Series([['Sem Categoria']] * len(metadata_df))
    
    df = pd.merge(ratings_df, metadata_df, on='Title', how='left')

    df['authors'] = df['authors'].apply(lambda d: d if isinstance(d, list) else ['Autor Desconhecido'])
    df['categories'] = df['categories'].apply(lambda d: d if isinstance(d, list) else ['Sem Categoria'])

    df['full_review_text'] = df['summary'] + '. ' + df['text']
    df['cleaned_review_text'] = df['full_review_text'].apply(clean_review_text)
    df['review_time'] = pd.to_datetime(df['time'], unit='s')
    
    df['review_length'] = df['cleaned_review_text'].apply(lambda x: len(x.split()))
    
    user_stats = df.groupby('User_id').agg(
        user_review_count=('Title', 'count'),
        user_avg_score=('score', 'mean')
    ).reset_index()
    
    df = pd.merge(df, user_stats, on='User_id', how='left')
    
    return df