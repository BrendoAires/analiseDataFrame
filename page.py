import streamlit as st
import pandas as pd
import cohere
from dotenv import load_dotenv
import os

# Carregar a chave da API do Cohere do arquivo .env
load_dotenv()  # Carrega o arquivo .env
api_key = os.getenv("COHERE_API_KEY")

# Título do aplicativo
st.title("Análise de Dados com Cohere e Streamlit")

# Sidebar para upload de arquivos
with (st.sidebar):
    uploaded_file = st.file_uploader("Escolha um arquivo (CSV ou Excel)", type=["csv", "xlsx"])

    cohere_api_key = api_key

    # Título da aplicação
    st.title("Chatbot")

    # Inicializar sessão de mensagens
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você?"}]

    # Exibir mensagens existentes
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Entrada do usuário
    if prompt := st.chat_input("Digite sua mensagem..."):
        # Adicionar mensagem do usuário ao estado
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if not cohere_api_key:
            st.info("Por favor, insira sua chave da API Cohere para continuar.")
            st.stop()

        # Cliente Cohere
        co = cohere.Client(cohere_api_key)

        conversation = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
        )

        # Enviar mensagens para o Cohere utilizando o método generate
        response = co.generate(
            model="command",  # Substitua por "command-r-plus" ou outro modelo adequado
            prompt=conversation,
            max_tokens=300,
            temperature=0.7
        )

        # Processar resposta e exibir
        assistant_response = response.generations[0].text.strip()

        # Adicionar resposta do assistente à conversa
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.chat_message("assistant").write(assistant_response)

# Se um arquivo foi carregado
if uploaded_file is not None:
    # Exibir o nome do arquivo carregado
    st.write(f"**Arquivo carregado:** {uploaded_file.name}")

    # Identificar o tipo de arquivo e carregar os dados
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        # Exibir os dados no Streamlit
        st.subheader("Pré-visualização dos Dados")
        st.dataframe(df.head(10))

        # Estatísticas básicas
        st.subheader("Resumo Estatístico")
        st.write(df.describe())

        # Inicializar cliente Cohere
        co = cohere.Client(api_key)

        # Gerar uma amostra dos dados e estatísticas básicas
        sample_data = df.head(10).to_string(index=False)
        stats = df.describe().to_string()

        # Mensagem estruturada para análise
        prompt = f"""
        Você é um analista de dados. Aqui está uma amostra dos dados e as estatísticas básicas do arquivo que carregamos. 
        Por favor, analise os dados e forneça insights úteis.

        ### Amostra dos Dados:
        {sample_data}

        ### Estatísticas Básicas:
        {stats}

        ### Objetivo:
        Queremos insights relacionados a tendências, discrepâncias ou outras observações interessantes nos dados. Por favor, forneça recomendações ou análises detalhadas.
        """

        # Enviar para o modelo Cohere utilizando o método generate
        response = co.generate(
            model="command-r-plus-08-2024",  # Escolha o modelo adequado
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )

        # Exibir os insights gerados
        st.subheader("Insights Gerados pelo Modelo Cohere")
        insights = response.generations[0].text.strip()

        st.write(insights)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("Por favor, faça o upload de um arquivo para começar.")
