import streamlit as st
import requests
import time
from streamlit_lottie import st_lottie
from src.data_processing import load_and_prepare_data
from src.llm_integration import load_base_model_and_tokenizer, create_langchain_llm
from src.analysis import generate_book_analysis_langchain, generate_follow_up_answer

st.set_page_config(page_title="Dashboard de Insights da Editora", page_icon="📚", layout="wide")

@st.cache_data
def cached_load_data():
    return load_and_prepare_data(use_sample=True) 

@st.cache_resource
def cached_load_llm():
    model, tokenizer = load_base_model_and_tokenizer()
    llm = create_langchain_llm(model, tokenizer)
    return llm

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

if 'data_loaded' not in st.session_state:
    st.header("📚 Bem-vindo(a) ao Dashboard de Insights da Editora", anchor=False)
    st.markdown("---")
    
    lottie_url = "https://lottie.host/890b53c7-2484-486a-8488-ab91223513b3/u9C3GA3f6T.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, speed=1, width=250, height=250)

    with st.status("Iniciando processo...", expanded=True) as status:
        status.update(label="Carregando e processando avaliações...")
        df = cached_load_data()
        st.session_state['data'] = df
        
        status.update(label="Preparando modelo de Inteligência Artificial...")
        llm = cached_load_llm() #Usar a GPU
        st.session_state['llm'] = llm
        
        status.update(label="Tudo pronto! Bem-vindo(a) ao Dashboard.", state="complete", expanded=False)

    time.sleep(1) 
    st.session_state['data_loaded'] = True
    st.session_state.analysis_results = None
    st.session_state.messages = []
    st.rerun()

else:
    df = st.session_state['data']
    llm = st.session_state['llm']

    st.title("📚 Dashboard de Insights da Editora")
    st.markdown("Uma ferramenta para automatizar a análise de avaliações de livros.")

    st.sidebar.header("Filtros de Análise")
    author_placeholder = "--Selecione um Autor--"
    category_placeholder = "--Selecione uma Categoria--"

    def clear_filters():
        st.session_state["author_selector"] = author_placeholder
        st.session_state["category_selector"] = category_placeholder
        st.session_state.analysis_results = None 
        st.session_state.messages = []

    all_authors = sorted(df.explode('authors')['authors'].unique())
    author_options = [author_placeholder] + all_authors
    selected_author = st.sidebar.selectbox("1. Selecione um Autor (Opcional)", options=author_options, key="author_selector")

    if selected_author != author_placeholder:
        author_df = df[df['authors'].apply(lambda authors_list: selected_author in authors_list)]
        category_options = [category_placeholder] + sorted(author_df.explode('categories')['categories'].unique())
    else:
        all_categories = sorted(df.explode('categories')['categories'].unique())
        category_options = [category_placeholder] + all_categories
        
    selected_category = st.sidebar.selectbox("2. Selecione uma Categoria (Opcional)", options=category_options, key="category_selector")
    st.sidebar.button("Limpar Filtros", on_click=clear_filters)
    
    tab1, tab2, tab3 = st.tabs(["Análise de Livros", "Análise de Usuários", "Análise de Gêneros"])

    with tab1:
        st.header("Livros Encontrados")
        
        filtered_df = df.copy()
        if selected_author != author_placeholder:
            filtered_df = filtered_df[filtered_df['authors'].apply(lambda authors_list: selected_author in authors_list)]
        if selected_category != category_placeholder:
            filtered_df = filtered_df[filtered_df['categories'].apply(lambda cat_list: selected_category in cat_list)]

        if not filtered_df.empty and (selected_author != author_placeholder or selected_category != category_placeholder):
            
            agg_df = filtered_df.copy()
            agg_df['authors_str'] = agg_df['authors'].apply(lambda x: ', '.join(map(str, x)))
            agg_df['categories_str'] = agg_df['categories'].apply(lambda x: ', '.join(map(str, x)))
            
            summary_df = agg_df.groupby(['Title', 'authors_str']).agg(
                categories=('categories_str', lambda s: ', '.join(s.unique())),
                avg_score=('score', 'mean'),
                num_reviews=('Id', 'count')
            ).reset_index().sort_values(by='num_reviews', ascending=False)
            
            st.dataframe(summary_df.rename(columns={
                'Title': 'Título', 'authors_str': 'Autores', 'categories': 'Categorias', 
                'avg_score': 'Nota Média', 'num_reviews': 'Nº de Avaliações'
            }))
            
            st.markdown("---")
            st.subheader("Análise Detalhada com IA")

            selected_book_for_analysis = st.selectbox(
                "Escolha um livro da lista acima para analisar:",
                options=summary_df['Title'].unique()
            )

            if st.button(f"Analisar '{selected_book_for_analysis}'"):
                st.session_state.messages = [] 
                st.session_state.current_book_title = selected_book_for_analysis
                with st.spinner(f"A IA está analisando as avaliações..."):
                    book_df = df[df['Title'] == selected_book_for_analysis].copy()
                    st.session_state.analysis_results = generate_book_analysis_langchain(book_df, selected_book_for_analysis, llm)
            
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results
                current_book_title = st.session_state.current_book_title
                
                st.divider()
                st.subheader(f"Resultados para: {current_book_title}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Nota Média", f"{results.get('avg_score', 0):.2f} / 5.0")
                col2.metric("Nº de Avaliações", f"{results.get('num_reviews', 0)}")
                col3.metric("Polarização", f"{results.get('score_std_dev', 0):.2f}", help="Desvio padrão das notas. Valores altos (>1.2) indicam opiniões divididas.")
                col4.metric("Experiência do Leitor", f"{results.get('avg_reviewer_experience', 0):.0f}", help="Média de avaliações feitas pelos leitores deste livro.")

                st.divider()
                summary_dict = results.get('llm_summary')
                if summary_dict:
                    st.success("**Destaques Positivos**")
                    st.markdown(summary_dict.get('destaques_positivos', 'N/A'))
                    st.warning("**Críticas Construtivas**")
                    st.markdown(summary_dict.get('criticas_construtivas', 'N/A'))
                    st.info("**Insight Acionável**")
                    st.markdown(summary_dict.get('insight_acionavel', 'N/A'))

                st.divider()
                st.subheader("💬 Converse com os Dados")
                
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if user_question := st.chat_input(f"Faça uma pergunta sobre '{current_book_title}'..."):
                    st.session_state.messages.append({"role": "user", "content": user_question})
                    with st.chat_message("user"):
                        st.markdown(user_question)

                    with st.chat_message("assistant"):
                        with st.spinner("Pensando..."):
                            book_df = df[df['Title'] == current_book_title].copy()
                            response = generate_follow_up_answer(book_df, current_book_title, user_question, llm)
                            st.markdown(response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

        elif selected_author == author_placeholder and selected_category == category_placeholder:
            st.info("Utilize os filtros na barra lateral para começar a explorar os livros.")
        else:
            st.warning("Nenhum livro encontrado para a combinação de filtros selecionada.")

    with tab2:
        st.header("🔎 Buscador de Usuários Relevantes")
        st.markdown("Encontre os usuários mais ativos, os maiores fãs ou os críticos mais severos.")
        
        sort_by = st.selectbox(
            "Classificar usuários por:",
            options=["Maior Número de Avaliações", "Maiores Fãs (Melhor Nota Média)", "Maiores Críticos (Pior Nota Média)"]
        )
        unique_users_df = df.drop_duplicates(subset=['User_id']).copy()
        if sort_by == "Maior Número de Avaliações":
            top_users = unique_users_df.nlargest(20, 'user_review_count')
        elif sort_by == "Maiores Fãs (Melhor Nota Média)":
            top_users = unique_users_df.nlargest(20, 'user_avg_score')
        else:
            top_users = unique_users_df.nsmallest(20, 'user_avg_score')
        st.subheader(f"Top 20 Usuários: {sort_by}")
        st.dataframe(
            top_users[['profileName', 'user_review_count', 'user_avg_score']],
            column_config={"profileName": "Nome do Perfil", "user_review_count": "Qtd. de Avaliações", "user_avg_score": "Nota Média Concedida"},
            use_container_width=True
        )

    with tab3:

        st.header("📊 Análise de Gêneros Literários")
        st.markdown("Explore insights sobre desempenho e características dos diferentes gêneros/categorias de livros.")
        
        st.subheader("Visão Geral dos Gêneros")
        
        exploded_categories = df.explode('categories')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            unique_genres = exploded_categories['categories'].nunique()
            st.metric("Total de Gêneros Únicos", unique_genres)
        
        with col2:
            avg_books_per_genre = round(exploded_categories['categories'].value_counts().mean(), 1)
            st.metric("Média de Livros por Gênero", avg_books_per_genre)
        
        with col3:
            avg_score_per_genre = round(exploded_categories.groupby('categories')['score'].mean().mean(), 2)
            st.metric("Nota Média Geral dos Gêneros", f"{avg_score_per_genre}/5.0")
        
        st.subheader("Ranking de Gêneros")
        
        metric_choice = st.selectbox(
            "Classificar gêneros por:",
            options=["Maior Número de Livros", "Melhor Avaliação Média", "Maior Número de Avaliações"],
            key="genre_metric"
        )
        
        if metric_choice == "Maior Número de Livros":
            genre_stats = exploded_categories['categories'].value_counts().reset_index()
            genre_stats.columns = ['Gênero', 'Número de Livros']
        elif metric_choice == "Melhor Avaliação Média":
            genre_stats = exploded_categories.groupby('categories')['score'].mean().sort_values(ascending=False).reset_index()
            genre_stats.columns = ['Gênero', 'Nota Média']
        else:
            genre_stats = exploded_categories.groupby('categories')['Id'].count().sort_values(ascending=False).reset_index()
            genre_stats.columns = ['Gênero', 'Total de Avaliações']
        
        st.dataframe(
            genre_stats.head(20),
            use_container_width=True,
            height=400
        )
        
        st.subheader("Análise Detalhada por Gênero")
        
        selected_genre = st.selectbox(
            "Selecione um gênero para análise detalhada:",
            options=sorted(exploded_categories['categories'].unique()),
            key="genre_selector"
        )
        
        if selected_genre:
            genre_df = exploded_categories[exploded_categories['categories'] == selected_genre]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Livros", genre_df['Title'].nunique())
            with col2:
                st.metric("Total de Avaliações", len(genre_df))
            with col3:
                avg_score = genre_df['score'].mean()
                st.metric("Nota Média", f"{avg_score:.2f}/5.0")
            with col4:
                st.metric("Autores Diferentes", genre_df['authors'].explode().nunique())
            
            st.divider()
            
            st.subheader(f"Top Livros do Gênero '{selected_genre}'")
            
            top_books = genre_df.groupby('Title').agg(
                Autores=('authors', lambda x: ', '.join(set([a for sublist in x for a in sublist]))),
                Nota_Média=('score', 'mean'),
                N_Avaliações=('Id', 'count')
            ).sort_values('N_Avaliações', ascending=False).head(10)
            
            st.dataframe(
                top_books.reset_index(),
                column_config={
                    "Title": "Título",
                    "Autores": "Autores",
                    "Nota_Média": "Nota Média",
                    "N_Avaliações": "Nº de Avaliações"
                },
                use_container_width=True
            )
                    