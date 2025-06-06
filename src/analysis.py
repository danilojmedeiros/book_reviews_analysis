import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

def generate_book_analysis_langchain(book_df: pd.DataFrame, book_title: str, llm):
    if book_df.empty:
        return {"avg_score": 0, "num_reviews": 0, "avg_review_length": 0, 
                "score_distribution": pd.Series(dtype='int64'), "score_std_dev": 0, 
                "avg_reviewer_experience": 0, "llm_summary": None}
    
    avg_score = book_df['score'].mean()
    num_reviews = len(book_df)
    avg_review_length = book_df['review_length'].mean()
    score_std_dev = book_df['score'].std()
    avg_reviewer_experience = book_df['user_review_count'].mean()

    response_schemas = [
        ResponseSchema(name="destaques_positivos", description="Uma lista em bullet points dos principais elogios."),
        ResponseSchema(name="criticas_construtivas", description="Uma lista em bullet points das principais críticas."),
        ResponseSchema(name="insight_acionavel", description="Uma sugestão de ação concreta para a editora.")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    template = """User: Você é um analista de mercado editorial especialista. Sua tarefa é analisar uma coleção de avaliações de usuários para o livro intitulado "{book_title}". Com base nas seguintes avaliações, forneça um resumo conciso e acionável para um executivo da editora.

    AVALIAÇÕES:
    ---
    {reviews_text}
    ---

    Por favor, estruture sua resposta exatamente no seguinte formato JSON, sem nenhum texto adicional antes ou depois do JSON:
    {format_instructions}

    Assistant:"""

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["book_title", "reviews_text"],
        partial_variables={"format_instructions": format_instructions}
    )

    reviews_to_analyze = "\n".join("- " + review for review in book_df['cleaned_review_text'].head(20))
    input_data = {"book_title": book_title, "reviews_text": reviews_to_analyze}

    # Retorna esse exemplo em (llm=None)
    if not llm:
        llm_summary_dict = {
            "destaques_positivos": "- O enredo é acelerado e envolvente.\n- Os personagens são bem desenvolvidos.",
            "criticas_construtivas": "- O meio do livro tem um ritmo mais lento.",
            "insight_acionavel": "Focar o marketing no ritmo da trama para atrair leitores."
        }
    else:
        try:
            chain = prompt_template | llm | output_parser
            llm_summary_dict = chain.invoke(input_data)
        except Exception as e:
            print(f"Erro ao invocar a chain do LangChain: {e}")
            llm_summary_dict = {
                "destaques_positivos": "Erro ao gerar análise.",
                "criticas_construtivas": "O modelo pode não ter respondido no formato esperado.",
                "insight_acionavel": "Verifique os logs para mais detalhes."
            }
    
    return {
        "avg_score": avg_score, 
        "num_reviews": num_reviews, 
        "avg_review_length": avg_review_length,
        "score_std_dev": score_std_dev,
        "avg_reviewer_experience": avg_reviewer_experience,
        "llm_summary": llm_summary_dict
    }

def generate_follow_up_answer(book_df: pd.DataFrame, book_title: str, question: str, llm):
    if not llm or book_df.empty or not question:
        return "Por favor, selecione um livro e faça uma pergunta."

    reviews_context = "\n".join("- " + review for review in book_df['cleaned_review_text'].head(50))

    follow_up_template = """User: Você é um assistente de análise de dados. Sua tarefa é responder à **Pergunta do Usuário** sobre o livro **"{book_title}"**, baseando-se estritamente no **Contexto das Avaliações** fornecido. Seja direto e limite sua resposta a no máximo 60 palavras.

    **Contexto das Avaliações:**
    ---
    {reviews_context}
    ---

    **Pergunta do Usuário:** {user_question}

    Assistant:"""

    prompt = PromptTemplate.from_template(follow_up_template)
    chain = prompt | llm 

    try:
        response = chain.invoke({
            "book_title": book_title, # Passa o título para o prompt
            "reviews_context": reviews_context,
            "user_question": question
        })
        return response
    except Exception as e:
        print(f"Erro na pergunta de acompanhamento: {e}")
        return "Ocorreu um erro ao processar sua pergunta."