# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import os
# Novas importações para modelos adicionais
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Configuração da página
st.set_page_config(
    page_title="CreditWise - Credit Scoring Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para criar dados sintéticos de clientes
def generate_synthetic_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    # Características
    idade = np.random.randint(18, 75, n_samples)
    renda_anual = np.random.normal(60000, 30000, n_samples)
    tempo_emprego = np.random.randint(0, 40, n_samples)
    tempo_residencia = np.random.randint(0, 20, n_samples)
    qtd_contas = np.random.randint(0, 10, n_samples)
    divida_total = np.random.normal(20000, 15000, n_samples)
    taxa_utilizacao_credito = np.clip(np.random.normal(0.5, 0.3, n_samples), 0, 1)
    consultas_credito = np.random.randint(0, 10, n_samples)
    atrasos_pagamento = np.random.randint(0, 12, n_samples)
    historico_pagamento = np.random.randint(0, 100, n_samples)
    
    # Gerando a variável alvo (bom pagador vs mau pagador)
    # Criando uma relação lógica entre as características e o target
    probabilidade = (
        - 0.01 * atrasos_pagamento 
        + 0.005 * historico_pagamento 
        - 0.5 * taxa_utilizacao_credito
        + 0.0000015 * renda_anual
        + 0.01 * tempo_emprego
        - 0.05 * consultas_credito
    )
    
    probabilidade = 1 / (1 + np.exp(-probabilidade))  # Transformação logística
    target = (np.random.random(n_samples) < probabilidade).astype(int)
    
    # Criando o DataFrame
    data = pd.DataFrame({
        'idade': idade,
        'renda_anual': renda_anual,
        'tempo_emprego': tempo_emprego,
        'tempo_residencia': tempo_residencia,
        'qtd_contas': qtd_contas,
        'divida_total': divida_total,
        'taxa_utilizacao_credito': taxa_utilizacao_credito,
        'consultas_credito': consultas_credito,
        'atrasos_pagamento': atrasos_pagamento,
        'historico_pagamento': historico_pagamento,
        'bom_pagador': target
    })
    
    return data

# Função para gerar dados sintéticos de cobrança
def generate_collection_data(data, random_state=42):
    np.random.seed(random_state)
    
    # Filtrar apenas clientes maus pagadores
    bad_payers = data[data['bom_pagador'] == 0].copy()
    
    if len(bad_payers) == 0:
        # Se não houver maus pagadores, crie alguns artificialmente
        bad_payers = data.copy().iloc[:100]
    
    # Adicionar informações de cobrança
    n_samples = len(bad_payers)
    
    # Dias de atraso
    dias_atraso = np.random.randint(10, 180, n_samples)
    
    # Valor em atraso
    valor_atraso = np.random.normal(5000, 3000, n_samples)
    valor_atraso = np.clip(valor_atraso, 500, 20000)
    
    # Número de contatos realizados
    contatos_realizados = np.random.randint(0, 10, n_samples)
    
    # Resposta a contatos (0: sem resposta, 1: respondeu)
    resposta_contato = np.random.binomial(1, 0.6, n_samples)
    
    # Estratégia de cobrança (aleatória)
    estrategias = ['SMS', 'E-mail', 'Ligação', 'Carta', 'Visita']
    estrategia_cobranca = np.random.choice(estrategias, n_samples)
    
    # Proposta de negociação (0: não feita, 1: feita)
    proposta_negociacao = np.random.binomial(1, 0.4, n_samples)
    
    # Resultado de cobrança (variável target)
    # 0: Não recuperado, 1: Negociado, 2: Pago integralmente
    base_proba = 0.3  # probabilidade base de sucesso
    
    # Modelo para definir resultado
    prob_ajuste = (
        - 0.001 * dias_atraso 
        + 0.02 * (contatos_realizados > 3)
        + 0.15 * resposta_contato
        + 0.25 * proposta_negociacao
        - 0.00005 * valor_atraso
    )
    
    # Ajustar pela estratégia
    estrategia_bonus = {
        'SMS': 0.05,
        'E-mail': 0.02,
        'Ligação': 0.15,
        'Carta': 0.08,
        'Visita': 0.25
    }
    
    for i, estrategia in enumerate(estrategia_cobranca):
        prob_ajuste[i] += estrategia_bonus[estrategia]
    
    # Calcular probabilidade final
    prob_sucesso = np.clip(base_proba + prob_ajuste, 0.01, 0.95)
    
    # Determinar resultados
    resultados = np.zeros(n_samples)
    for i in range(n_samples):
        rand = np.random.random()
        if rand < prob_sucesso[i]:
            if rand < prob_sucesso[i] * 0.7:  # 70% dos sucessos são negociações
                resultados[i] = 1  # Negociado
            else:
                resultados[i] = 2  # Pago integralmente
    
    # Criar DataFrame de cobrança
    collection_data = bad_payers.reset_index(drop=True).copy()
    collection_data['dias_atraso'] = dias_atraso
    collection_data['valor_atraso'] = valor_atraso
    collection_data['contatos_realizados'] = contatos_realizados
    collection_data['resposta_contato'] = resposta_contato
    collection_data['estrategia_cobranca'] = estrategia_cobranca
    collection_data['proposta_negociacao'] = proposta_negociacao
    collection_data['resultado'] = resultados
    
    # Adicionar informações úteis
    collection_data['resultado_texto'] = collection_data['resultado'].map({
        0: 'Não recuperado',
        1: 'Negociado',
        2: 'Pago integralmente'
    })
    
    collection_data['sucesso_cobranca'] = collection_data['resultado'] > 0
    
    return collection_data

# Função para treinar o modelo de score de crédito
def train_credit_scoring_model(data):
    # Divisão entre features e target
    X = data.drop('bom_pagador', axis=1)
    y = data['bom_pagador']
    
    # Divisão entre treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinamento do modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Avaliação do modelo
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calcular pontuações de crédito para o conjunto de teste
    credit_scores = model.predict_proba(X_test_scaled)[:, 1] * 1000
    
    return model, scaler, report, X_test, y_test, credit_scores

# Título principal
st.title("💳 CreditWise - Credit Scoring Engine")
st.markdown("### Sistema Inteligente de Avaliação de Crédito")

# Verificar se já existem dados ou gerar novos
if 'data' not in st.session_state:
    st.session_state.data = generate_synthetic_data(2000)
    st.session_state.show_welcome = True

if 'model' not in st.session_state or 'scaler' not in st.session_state:
    with st.spinner("Treinando modelo de credit scoring..."):
        model, scaler, report, X_test, y_test, credit_scores = train_credit_scoring_model(st.session_state.data)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.report = report
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.credit_scores = credit_scores

# Menu lateral
with st.sidebar:
    st.header("Menu")
    app_mode = st.selectbox(
        "Escolha a funcionalidade:",
        ["Visão Geral", "Explorar Dados", "Modelo de Crédito", "Comparação de Modelos", "Explicabilidade (SHAP)", "Simulador de Crédito", "Análise de Cobrança"]
    )
    
    st.divider()
    st.markdown("### Sobre o CreditWise")
    st.markdown("""
    O CreditWise é um sistema de scoring de crédito que utiliza
    machine learning para avaliar o risco de empréstimos. 
    
    Os dados utilizados são sintéticos e servem apenas para demonstração.
    """)

# Página inicial com visão geral
if app_mode == "Visão Geral":
    st.header("Bem-vindo ao CreditWise Credit Scoring Engine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("O que é Credit Scoring?")
        st.markdown("""
        Credit Scoring é um método estatístico usado por instituições financeiras para avaliar 
        a probabilidade de um cliente não pagar um empréstimo. Isso resulta em uma pontuação 
        numérica que representa o risco de crédito associado a um indivíduo.
        """)
        
        st.subheader("Como funciona o CreditWise?")
        st.markdown("""
        1. **Coleta de Dados**: Informações financeiras e comportamentais do cliente
        2. **Pré-processamento**: Normalização e tratamento dos dados
        3. **Modelagem**: Algoritmos de machine learning para prever comportamento
        4. **Scoring**: Conversão da probabilidade em uma pontuação de 0-1000
        5. **Decisão**: Aprovação ou recusa com base em políticas de crédito
        """)
    
    with col2:
        # Gráfico de distribuição de scores
        fig = px.histogram(
            pd.DataFrame({'credit_score': st.session_state.credit_scores}), 
            x='credit_score',
            nbins=50,
            title='Distribuição dos Credit Scores',
            labels={'credit_score': 'Credit Score', 'count': 'Frequência'},
            color_discrete_sequence=['#3366CC']
        )
        fig.update_layout(xaxis_range=[0, 1000])
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas de performance do modelo
        st.metric("Acurácia do modelo", f"{st.session_state.report['accuracy']:.2%}")
        
        good_payers = st.session_state.data['bom_pagador'].mean()
        st.metric("Taxa de bons pagadores", f"{good_payers:.2%}")

    # Exibir exemplos de clientes
    st.subheader("Exemplos de Clientes")
    sample_clients = st.session_state.data.sample(5)
    # Converter bom_pagador para string mais informativo
    sample_clients_display = sample_clients.copy()
    sample_clients_display['bom_pagador'] = sample_clients_display['bom_pagador'].map({1: 'Bom Pagador', 0: 'Mau Pagador'})
    st.dataframe(sample_clients_display)

# Explorar dados
elif app_mode == "Explorar Dados":
    st.header("Exploração e Análise de Dados")
    
    # Menu de análise
    analysis_type = st.radio(
        "Selecione o tipo de análise:",
        ["Estatísticas Descritivas", "Distribuições", "Correlações", "Análise Bivariada"],
        horizontal=True
    )
    
    if analysis_type == "Estatísticas Descritivas":
        st.subheader("Estatísticas Descritivas")
        
        # Estatísticas básicas
        desc_stats = st.session_state.data.describe().T
        
        # Adicionar contagem de bons e maus pagadores
        good_bad_counts = st.session_state.data['bom_pagador'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(desc_stats)
        
        with col2:
            # Gráfico de pizza para bons/maus pagadores
            labels = ['Mau Pagador', 'Bom Pagador']
            values = [good_bad_counts.get(0, 0), good_bad_counts.get(1, 0)]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig.update_layout(title_text='Distribuição de Bons e Maus Pagadores')
            st.plotly_chart(fig, use_container_width=True)
            
            # Informações adicionais
            st.markdown("### Observações:")
            st.markdown(f"- Total de registros: {len(st.session_state.data)}")
            st.markdown(f"- Proporção de bons pagadores: {values[1]/(values[0]+values[1]):.2%}")
            st.markdown(f"- Características disponíveis: {len(st.session_state.data.columns)-1}")
    
    elif analysis_type == "Distribuições":
        st.subheader("Distribuições das Variáveis")
        
        # Seletor de variável
        numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_cols.remove('bom_pagador')  # Remover a variável alvo
        
        selected_var = st.selectbox("Selecione a variável para visualizar:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma
            fig = px.histogram(
                st.session_state.data, 
                x=selected_var,
                color="bom_pagador",
                barmode="overlay",
                color_discrete_map={0: "#FF4B4B", 1: "#2EB086"},
                labels={"bom_pagador": "Status"},
                category_orders={"bom_pagador": [0, 1]},
                title=f"Distribuição de {selected_var} por status de pagamento"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(
                st.session_state.data, 
                x="bom_pagador", 
                y=selected_var,
                color="bom_pagador",
                color_discrete_map={0: "#FF4B4B", 1: "#2EB086"},
                labels={"bom_pagador": "Status", selected_var: selected_var},
                category_orders={"bom_pagador": [0, 1]},
                title=f"Box Plot de {selected_var} por status de pagamento"
            )
            fig.update_xaxes(ticktext=["Mau Pagador", "Bom Pagador"], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas por grupo
        st.subheader("Estatísticas por Grupo")
        group_stats = st.session_state.data.groupby('bom_pagador')[selected_var].describe().reset_index()
        group_stats['bom_pagador'] = group_stats['bom_pagador'].map({0: 'Mau Pagador', 1: 'Bom Pagador'})
        st.dataframe(group_stats)
    
    elif analysis_type == "Correlações":
        st.subheader("Matriz de Correlação")
        
        # Calcular correlações
        corr_matrix = st.session_state.data.corr()
        
        # Gerar heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            annot=True,
            fmt=".2f"
        )
        plt.title('Matriz de Correlação', fontsize=16)
        st.pyplot(fig)
        
        # Correlações com a variável alvo
        st.subheader("Correlações com status de pagamento")
        target_corr = corr_matrix['bom_pagador'].drop('bom_pagador').sort_values(ascending=False)
        
        fig = px.bar(
            x=target_corr.index, 
            y=target_corr.values,
            labels={'x': 'Característica', 'y': 'Correlação'},
            title='Correlação com Status de Pagamento',
            color=target_corr.values,
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Explicação das correlações"):
            st.markdown("""
            - **Correlação positiva**: Um aumento na variável está associado a uma maior chance de ser um bom pagador.
            - **Correlação negativa**: Um aumento na variável está associado a uma menor chance de ser um bom pagador.
            - **Correlação próxima de zero**: A variável tem pouca ou nenhuma relação linear com o status de pagamento.
            """)
    
    elif analysis_type == "Análise Bivariada":
        st.subheader("Análise Bivariada")
        
        # Seleção de variáveis
        numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_cols.remove('bom_pagador')  # Remover a variável alvo
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Variável X:", numeric_cols, index=0)
        
        with col2:
            y_var = st.selectbox("Variável Y:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        
        # Scatter plot
        fig = px.scatter(
            st.session_state.data,
            x=x_var,
            y=y_var,
            color="bom_pagador",
            color_discrete_map={0: "#FF4B4B", 1: "#2EB086"},
            labels={x_var: x_var, y_var: y_var, "bom_pagador": "Status"},
            title=f"Relação entre {x_var} e {y_var} por status de pagamento",
            opacity=0.7
        )
        
        # Adicionar linhas de tendência
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas de correlação
        corr_value = st.session_state.data[[x_var, y_var]].corr().iloc[0, 1]
        
        st.metric("Correlação de Pearson", f"{corr_value:.4f}")
        
        if abs(corr_value) > 0.7:
            st.warning("Há uma forte correlação entre essas variáveis, o que pode indicar multicolinearidade.")
        
        # Quadrantes e distribuições marginais
        with st.expander("Ver distribuições marginais"):
            fig = px.scatter(
                st.session_state.data,
                x=x_var,
                y=y_var,
                color="bom_pagador",
                color_discrete_map={0: "#FF4B4B", 1: "#2EB086"},
                marginal_x="histogram",
                marginal_y="histogram"
            )
            st.plotly_chart(fig, use_container_width=True)

# Modelo de Crédito
elif app_mode == "Modelo de Crédito":
    st.header("Modelo de Credit Scoring")
    
    tab1, tab2, tab3 = st.tabs(["Performance do Modelo", "Importância das Variáveis", "Distribuição de Scores"])
    
    with tab1:
        st.subheader("Métricas de Performance")
        
        # Calcular matriz de confusão
        y_pred = st.session_state.model.predict(st.session_state.scaler.transform(st.session_state.X_test))
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Métricas principais
            st.metric("Acurácia", f"{st.session_state.report['accuracy']:.2%}")
            st.metric("Precisão (Classe Positiva)", f"{st.session_state.report['1']['precision']:.2%}")
            st.metric("Recall (Classe Positiva)", f"{st.session_state.report['1']['recall']:.2%}")
            st.metric("F1-Score (Classe Positiva)", f"{st.session_state.report['1']['f1-score']:.2%}")
        
        with col2:
            # Matriz de confusão
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo']
            )
            plt.title('Matriz de Confusão')
            plt.xlabel('Previsto')
            plt.ylabel('Real')
            st.pyplot(fig)
        
        # Curva ROC
        st.subheader("Curva ROC")
        
        y_prob = st.session_state.model.predict_proba(st.session_state.scaler.transform(st.session_state.X_test))[:, 1]
        fpr, tpr, _ = roc_curve(st.session_state.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = px.area(
            x=fpr, y=tpr,
            title=f'Curva ROC (AUC = {roc_auc:.2f})',
            labels=dict(x='Taxa de Falsos Positivos', y='Taxa de Verdadeiros Positivos'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Importância das Variáveis")
        
        # Extrair importância das variáveis do modelo
        if hasattr(st.session_state.model, 'feature_importances_'):
            # Para modelos baseados em árvores (Random Forest, XGBoost, etc.)
            importances = st.session_state.model.feature_importances_
            feature_names = st.session_state.X_test.columns
            
            # Criar DataFrame de importâncias
            importance_df = pd.DataFrame({
                'Variável': feature_names,
                'Importância': importances
            }).sort_values('Importância', ascending=False)
            
            # Visualizar importância das variáveis
            fig = px.bar(
                importance_df,
                x='Importância',
                y='Variável',
                orientation='h',
                title='Importância das Variáveis',
                color='Importância',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de importância
            st.write("### Ranking de Importância")
            importance_df['Importância (%)'] = importance_df['Importância'] * 100
            importance_df['Importância (%)'] = importance_df['Importância (%)'].apply(lambda x: f"{x:.2f}%")
            st.dataframe(importance_df[['Variável', 'Importância (%)']])
            
            # Análise descritiva das principais variáveis
            st.write("### Análise das Principais Variáveis")
            top_features = importance_df['Variável'].head(3).tolist()
            
            for feature in top_features:
                st.write(f"#### {feature}")
                
                fig = px.histogram(
                    st.session_state.data, 
                    x=feature,
                    color="bom_pagador",
                    barmode="overlay",
                    color_discrete_map={0: "#FF4B4B", 1: "#2EB086"},
                    labels={"bom_pagador": "Status"},
                    category_orders={"bom_pagador": [0, 1]},
                    title=f"Distribuição de {feature} por status de pagamento"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Para modelos sem feature_importances_ (regressão logística, redes neurais, etc.)
            st.info("Este modelo não suporta visualização direta de importância de variáveis. Considere usar SHAP para análise de importância.")
    
    with tab3:
        st.subheader("Distribuição de Scores")
        
        # Histograma de distribuição de scores
        fig = px.histogram(
            pd.DataFrame({'credit_score': st.session_state.credit_scores}),
            x='credit_score',
            nbins=50,
            title='Distribuição dos Credit Scores',
            labels={'credit_score': 'Credit Score', 'count': 'Frequência'},
            color_discrete_sequence=['#3366CC']
        )
        fig.update_layout(xaxis_range=[0, 1000])
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribuição por status de pagamento
        st.write("### Distribuição por Status de Pagamento")
        
        # Criar DataFrame com scores e status real
        score_status_df = pd.DataFrame({
            'credit_score': st.session_state.credit_scores,
            'bom_pagador': st.session_state.y_test.values
        })
        
        fig = px.histogram(
            score_status_df,
            x='credit_score',
            color='bom_pagador',
            barmode='overlay',
            nbins=50,
            opacity=0.7,
            title='Distribuição de Scores por Status',
            labels={'credit_score': 'Credit Score', 'count': 'Frequência', 'bom_pagador': 'Status'},
            color_discrete_map={0: "#FF4B4B", 1: "#2EB086"}
        )
        fig.update_layout(xaxis_range=[0, 1000])
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise de percentis
        st.write("### Análise por Percentis")
        
        percentiles = [10, 25, 50, 75, 90]
        score_percentiles = np.percentile(st.session_state.credit_scores, percentiles)
        
        percentile_df = pd.DataFrame({
            'Percentil': [f"{p}%" for p in percentiles],
            'Score': score_percentiles
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(percentile_df)
        
        with col2:
            # Definição das categorias de risco
            st.write("### Categorias de Risco")
            
            risk_categories = pd.DataFrame({
                'Categoria': ['Muito Alto Risco', 'Alto Risco', 'Médio Risco', 'Baixo Risco', 'Muito Baixo Risco'],
                'Faixa de Score': ['0-300', '301-500', '501-700', '701-850', '851-1000']
            })
            
            st.dataframe(risk_categories)

# Explicabilidade (SHAP)
elif app_mode == "Explicabilidade (SHAP)":
    st.header("Explicabilidade do Modelo com SHAP")
    
    st.write("""
    O SHAP (SHapley Additive exPlanations) é uma técnica para explicar as saídas de qualquer modelo de machine learning.
    Baseado na teoria dos jogos, SHAP usa valores de Shapley para atribuir a cada característica sua contribuição para a previsão.
    """)
    
    # Calcular valores SHAP (limitando a um subconjunto para performance)
    if 'shap_values' not in st.session_state:
        with st.spinner("Calculando valores SHAP (pode demorar um pouco)..."):
            # Usar uma amostra dos dados para cálculo de SHAP (por eficiência)
            sample_size = min(len(st.session_state.X_test), 100)  # Reduzindo para 100 para melhor performance
            X_sample = st.session_state.X_test.sample(sample_size, random_state=42)
            X_sample_scaled = st.session_state.scaler.transform(X_sample)
            
            # Criar o explainer e calcular valores SHAP
            try:
                # Usar o explainer adequado para o modelo
                if hasattr(st.session_state.model, 'feature_importances_'):
                    # Para modelos baseados em árvore (como Random Forest, XGBoost)
                    explainer = shap.TreeExplainer(st.session_state.model)
                    # Calcular valores SHAP para a classe positiva (bom pagador)
                    shap_values = explainer.shap_values(X_sample_scaled)
                    
                    # Para árvores com valores para ambas as classes
                    if isinstance(shap_values, list):
                        # Guardar no state 
                        st.session_state.shap_values = shap_values
                        st.session_state.shap_output_dim = 'multiclass'
                        st.session_state.shap_positive_class_idx = 1  # Índice da classe positiva
                    else:
                        # Caso em que o modelo retorna apenas uma classe
                        st.session_state.shap_values = shap_values
                        st.session_state.shap_output_dim = 'single'
                else:
                    # Para outros tipos de modelo (regressão logística, etc.)
                    # Aqui optamos por prever probabilidades da classe positiva
                    def model_predict(data):
                        return st.session_state.model.predict_proba(data)[:, 1]
                    
                    explainer = shap.Explainer(model_predict, X_sample_scaled)
                    shap_values = explainer(X_sample_scaled)
                    
                    st.session_state.shap_values = shap_values
                    st.session_state.shap_output_dim = 'single'
                
                # Guardar outros dados no state
                st.session_state.X_sample = X_sample
                st.session_state.X_sample_scaled = X_sample_scaled
                st.session_state.explainer = explainer
                
            except Exception as e:
                st.error(f"Erro ao calcular valores SHAP: {e}")
                st.info("Algumas visualizações SHAP podem não estar disponíveis para este tipo de modelo.")
                st.session_state.shap_values = None
    
    # Opções de visualização SHAP
    shap_plot_type = st.radio(
        "Selecione o tipo de visualização SHAP:",
        ["Resumo (Global)", "Dependência", "Decisão Local", "Exemplo Individual"],
        horizontal=True
    )
    
    if 'shap_values' not in st.session_state or st.session_state.shap_values is None:
        st.warning("SHAP não disponível para este modelo. Tente outro modelo.")
    else:
        # Primeiro, vamos realizar diagnóstico dos valores SHAP
        # Isso nos ajudará a entender sua estrutura e mostrar informações úteis para depuração
        if shap_plot_type == "Resumo (Global)":
            st.subheader("Resumo dos Impactos das Variáveis (Global)")
            
            # Mostrar método alternativo diretamente (mais confiável)
            try:
                # Usar o método mais simples e confiável: feature importances do modelo
                fig, ax = plt.subplots(figsize=(10, 8))
                
                if hasattr(st.session_state.model, 'feature_importances_'):
                    # Para modelos baseados em árvore, usar feature importances diretas
                    importances = st.session_state.model.feature_importances_
                    feature_names = st.session_state.X_sample.columns
                    
                    # Criar DataFrame de importâncias e ordenar
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Criar gráfico de barras
                    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                    ax.set_title('Importância das Variáveis (Feature Importance)')
                    ax.set_xlabel('Importância')
                    ax.set_ylabel('Variável')
                    
                    # Adicionar anotações de porcentagem
                    total_importance = importances.sum()
                    for i, row in enumerate(importance_df.itertuples()):
                        percentage = (row.Importance / total_importance) * 100
                        ax.text(row.Importance + 0.01, i, f"{percentage:.1f}%", va='center')
                    
                    st.pyplot(fig)
                    
                    # Mostrar tabela detalhada
                    importance_df['Importância (%)'] = importance_df['Importance'].apply(lambda x: f"{(x/total_importance)*100:.2f}%")
                    st.dataframe(importance_df[['Feature', 'Importância (%)']])
                    
                else:
                    # Para outros tipos de modelos
                    st.info("Este modelo não suporta feature importances diretas. Calculando importância baseada nos valores SHAP.")
                    
                    # Calcular importância baseada nos valores absolutos médios de SHAP
                    if st.session_state.shap_output_dim == 'multiclass':
                        # Para modelos com múltiplas classes
                        shap_values_arr = np.abs(st.session_state.shap_values[st.session_state.shap_positive_class_idx])
                        mean_abs_shap = np.mean(shap_values_arr, axis=0)
                    else:
                        # Para modelos com saída única
                        try:
                            shap_values_arr = np.abs(st.session_state.shap_values.values)  
                            mean_abs_shap = np.mean(shap_values_arr, axis=0)
                        except:
                            shap_values_arr = np.abs(st.session_state.shap_values)
                            mean_abs_shap = np.mean(shap_values_arr, axis=0)
                    
                    # Criar DataFrame de importâncias
                    feature_names = st.session_state.X_sample.columns
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': mean_abs_shap
                    }).sort_values('Importance', ascending=False)
                    
                    # Criar gráfico de barras
                    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                    ax.set_title('Importância das Variáveis (Baseada em SHAP)')
                    ax.set_xlabel('Importância Média Absoluta (SHAP)')
                    ax.set_ylabel('Variável')
                    
                    # Adicionar anotações de porcentagem
                    total_importance = mean_abs_shap.sum()
                    for i, row in enumerate(importance_df.itertuples()):
                        percentage = (row.Importance / total_importance) * 100
                        ax.text(row.Importance + 0.01, i, f"{percentage:.1f}%", va='center')
                    
                    st.pyplot(fig)
                
                # Interpretação
                st.write("""
                ### Interpretação da Importância das Variáveis
                
                Este gráfico mostra a importância relativa de cada variável para o modelo:
                - Quanto maior a barra, mais impacto a variável tem nas previsões
                - Porcentagens indicam a contribuição relativa de cada variável
                
                Variáveis com maior importância devem ser monitoradas com mais atenção, pois têm maior impacto no score de crédito.
                """)
                
            except Exception as e:
                st.error(f"Erro ao gerar importância das variáveis: {e}")
                st.info("Visualização alternativa não disponível para este modelo.")
        
        elif shap_plot_type == "Dependência":
            st.subheader("Análise de Dependência SHAP")
            
            # Método simplificado para análise de dependência
            try:
                # Selecionar variável para análise de dependência
                feature_names = st.session_state.X_sample.columns.tolist()
                selected_feature = st.selectbox("Selecione uma variável para análise:", feature_names)
                
                # Encontrar o índice da feature selecionada
                feature_idx = feature_names.index(selected_feature)
                
                # Obter dados para a análise
                X_values = st.session_state.X_sample[selected_feature].values
                
                # Verificar se X_values tem forma adequada
                if len(X_values.shape) > 1:
                    X_values = X_values.flatten()
                
                # Aqui vamos diagnosticar a estrutura dos valores SHAP para depuração
                if st.session_state.shap_output_dim == 'multiclass':
                    st.write(f"Tipo de modelo: multiclass")
                    shap_values_arr = st.session_state.shap_values[st.session_state.shap_positive_class_idx]
                    st.write(f"Shape dos valores SHAP: {shap_values_arr.shape}")
                else:
                    st.write(f"Tipo de modelo: single")
                    try:
                        shap_values_arr = st.session_state.shap_values.values
                        st.write(f"Shape dos valores SHAP (values): {shap_values_arr.shape}")
                    except:
                        shap_values_arr = st.session_state.shap_values
                        st.write(f"Shape dos valores SHAP (direct): {shap_values_arr.shape}")
                
                st.write(f"Shape da feature X: {X_values.shape}")
                
                # Garantir que os valores SHAP têm o mesmo tamanho que os dados de entrada
                # Isso pode ocorrer quando os valores SHAP são calculados para múltiplas classes
                if st.session_state.shap_output_dim == 'multiclass':
                    # Para modelos com valores para múltiplas classes
                    shap_values_arr = st.session_state.shap_values[st.session_state.shap_positive_class_idx]
                    
                    # Verificar se é um array 2D com a dimensão correta
                    if len(shap_values_arr.shape) > 1:
                        if shap_values_arr.shape[0] == len(X_values) and shap_values_arr.shape[1] > feature_idx:
                            y_values = shap_values_arr[:, feature_idx]
                        else:
                            # Se as dimensões não correspondem, provavelmente temos uma incompatibilidade
                            st.error(f"Incompatibilidade de dimensões: SHAP shape {shap_values_arr.shape}, X shape {X_values.shape}")
                            st.warning("Tentando redimensionar para corresponder...")
                            
                            # Tentar pegar apenas os primeiros elementos para corresponder
                            if shap_values_arr.shape[0] > len(X_values):
                                shap_values_arr = shap_values_arr[:len(X_values)]
                            
                            if shap_values_arr.shape[1] > feature_idx:
                                y_values = shap_values_arr[:, feature_idx]
                            else:
                                st.error(f"Índice de feature {feature_idx} fora dos limites {shap_values_arr.shape[1]}")
                                st.stop()
                    else:
                        y_values = shap_values_arr  # Array 1D
                else:
                    # Para modelos com uma saída
                    try:
                        # Tente acessar o atributo .values (para objetos SHAP mais recentes)
                        shap_values_arr = st.session_state.shap_values.values
                        
                        # Verificar se é um array 2D com a dimensão correta
                        if len(shap_values_arr.shape) > 1:
                            if shap_values_arr.shape[0] == len(X_values) and shap_values_arr.shape[1] > feature_idx:
                                y_values = shap_values_arr[:, feature_idx]
                            else:
                                # Se as dimensões não correspondem, provavelmente temos uma incompatibilidade
                                st.error(f"Incompatibilidade de dimensões: SHAP shape {shap_values_arr.shape}, X shape {X_values.shape}")
                                st.warning("Tentando redimensionar para corresponder...")
                                
                                # Tentar pegar apenas os primeiros elementos para corresponder
                                if shap_values_arr.shape[0] > len(X_values):
                                    shap_values_arr = shap_values_arr[:len(X_values)]
                                
                                if shap_values_arr.shape[1] > feature_idx:
                                    y_values = shap_values_arr[:, feature_idx]
                                else:
                                    st.error(f"Índice de feature {feature_idx} fora dos limites {shap_values_arr.shape[1]}")
                                    st.stop()
                        else:
                            y_values = shap_values_arr  # Array 1D
                    except:
                        # Se não tiver .values, tente usar diretamente
                        shap_values_arr = st.session_state.shap_values
                        
                        # Verificar se é um array 2D com a dimensão correta
                        if len(shap_values_arr.shape) > 1:
                            if shap_values_arr.shape[0] == len(X_values) and shap_values_arr.shape[1] > feature_idx:
                                y_values = shap_values_arr[:, feature_idx]
                            else:
                                # Se as dimensões não correspondem, provavelmente temos uma incompatibilidade
                                st.error(f"Incompatibilidade de dimensões: SHAP shape {shap_values_arr.shape}, X shape {X_values.shape}")
                                st.warning("Tentando redimensionar para corresponder...")
                                
                                # Tentar pegar apenas os primeiros elementos para corresponder
                                if shap_values_arr.shape[0] > len(X_values):
                                    shap_values_arr = shap_values_arr[:len(X_values)]
                                
                                if len(shap_values_arr) > 0 and shap_values_arr.shape[1] > feature_idx:
                                    y_values = shap_values_arr[:, feature_idx]
                                else:
                                    st.error(f"Índice de feature {feature_idx} fora dos limites ou array vazio")
                                    st.stop()
                        else:
                            y_values = shap_values_arr  # Array 1D
                
                # Verificar se y_values tem forma adequada
                if hasattr(y_values, 'shape') and len(y_values.shape) > 1:
                    y_values = y_values.flatten()
                
                # Verificar se os arrays têm o mesmo tamanho
                if len(X_values) != len(y_values):
                    st.error(f"Erro: X tem {len(X_values)} elementos e y tem {len(y_values)} elementos.")
                    st.warning("Redimensionando arrays para corresponder ao menor tamanho...")
                    
                    # Garantir que os arrays tenham o mesmo tamanho
                    min_size = min(len(X_values), len(y_values))
                    X_values = X_values[:min_size]
                    y_values = y_values[:min_size]
                
                # Exibir tamanhos finais para depuração
                st.write(f"Tamanhos finais: X = {len(X_values)}, y = {len(y_values)}")
                
                # Criar scatter plot
                fig, ax = plt.subplots(figsize=(10, 7))
                
                # Usar cores para distinguir impacto positivo e negativo
                colors = ['#2EB086' if v > 0 else '#FF4B4B' for v in y_values]
                sizes = [30 + 20 * abs(v) for v in y_values]  # Tamanho varia com magnitude
                
                ax.scatter(X_values, y_values, c=colors, s=sizes, alpha=0.7)
                
                # Adicionar linha de tendência
                z = np.polyfit(X_values, y_values, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(X_values), max(X_values), 100)
                ax.plot(x_range, p(x_range), "k--", alpha=0.7, linewidth=2)
                
                # Adicionar linha horizontal em y=0
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                
                # Configurar gráfico
                ax.set_xlabel(selected_feature)
                ax.set_ylabel(f'Impacto SHAP para {selected_feature}')
                ax.set_title(f'Análise de Dependência para {selected_feature}')
                
                # Adicionar legendas e anotações
                ax.annotate(f"Tendência: y = {z[0]:.4f}x + {z[1]:.4f}", 
                           xy=(0.02, 0.95), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))
                
                # Exibir gráfico
                st.pyplot(fig)
                
                # Interpretação da dependência
                trend_direction = "positiva" if z[0] > 0 else "negativa"
                st.write(f"""
                ### Interpretação da Dependência para {selected_feature}
                
                Este gráfico mostra como valores diferentes de {selected_feature} impactam o score:
                
                - **Tendência {trend_direction}**: {'Valores maiores tendem a aumentar' if z[0] > 0 else 'Valores maiores tendem a reduzir'} o score
                - **Pontos verdes**: Valores que contribuem positivamente para o score
                - **Pontos vermelhos**: Valores que contribuem negativamente para o score
                - **Tamanho dos pontos**: Indica a magnitude da contribuição
                
                A linha pontilhada mostra a tendência geral da relação.
                """)
                
                # Adicionar estatísticas sobre a variável
                st.write("### Estatísticas da Variável")
                stats_df = pd.DataFrame({
                    'Estatística': ['Média', 'Mediana', 'Mínimo', 'Máximo', 'Desvio Padrão', 'Correlação com SHAP'],
                    'Valor': [
                        f"{X_values.mean():.2f}",
                        f"{np.median(X_values):.2f}",
                        f"{X_values.min():.2f}",
                        f"{X_values.max():.2f}",
                        f"{X_values.std():.2f}",
                        f"{np.corrcoef(X_values, y_values)[0,1]:.2f}"
                    ]
                })
                st.dataframe(stats_df)
                
            except Exception as e:
                st.error(f"Erro ao gerar análise de dependência: {e}")
                st.info("Tente selecionar outra variável ou utilizar outro tipo de visualização.")
        
        elif shap_plot_type == "Decisão Local":
            st.subheader("Explicação de Decisões Locais")
            
            try:
                # Selecionar amostra aleatória
                random_idx = st.slider("Selecione um exemplo:", 0, len(st.session_state.X_sample) - 1, 0)
                
                # Obter dados do exemplo
                example_data = st.session_state.X_sample.iloc[random_idx].to_frame().reset_index()
                example_data.columns = ['Característica', 'Valor']
                
                # Exibir dados do exemplo
                st.write("### Dados do Exemplo Selecionado")
                st.dataframe(example_data)
                
                # Calcular a previsão do modelo para este exemplo
                example_scaled = st.session_state.X_sample_scaled[random_idx].reshape(1, -1)
                example_prediction = st.session_state.model.predict_proba(example_scaled)[0, 1]
                example_class = 1 if example_prediction > 0.5 else 0
                
                # Exibir previsão
                pred_color = "#2EB086" if example_class == 1 else "#FF4B4B"
                pred_label = "Bom Pagador" if example_class == 1 else "Mau Pagador"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Probabilidade de Bom Pagador", 
                        f"{example_prediction:.2%}", 
                        delta=f"{example_prediction-0.5:.2%} vs. limiar 50%"
                    )
                with col2:
                    st.markdown(f"### Classificação: <span style='color:{pred_color}'>{pred_label}</span>", unsafe_allow_html=True)
                
                # Obter valores SHAP para o exemplo
                feature_names = st.session_state.X_sample.columns.tolist()
                
                try:
                    # Primeiramente, capturar o tipo e formato dos valores SHAP
                    if st.session_state.shap_output_dim == 'multiclass':
                        # Para modelos com múltiplas classes
                        shap_values_arr = st.session_state.shap_values[st.session_state.shap_positive_class_idx]
                        if len(shap_values_arr.shape) > 1:
                            shap_example = shap_values_arr[random_idx, :]
                        else:
                            shap_example = shap_values_arr[random_idx]
                    else:
                        # Para modelos com uma saída
                        try:
                            shap_values_arr = st.session_state.shap_values.values
                            if len(shap_values_arr.shape) > 1:
                                shap_example = shap_values_arr[random_idx, :]
                            else:
                                shap_example = shap_values_arr[random_idx]
                        except:
                            shap_values_arr = st.session_state.shap_values
                            if len(shap_values_arr.shape) > 1:
                                shap_example = shap_values_arr[random_idx, :]
                            else:
                                shap_example = shap_values_arr[random_idx]
                    
                    # Verificar se shap_example tem a forma correta e adaptá-la se necessário
                    if hasattr(shap_example, 'shape') and len(shap_example.shape) > 1:
                        shap_example = shap_example.flatten()
                    
                    # Criar DataFrame para visualização
                    shap_df = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP value': shap_example
                    }).sort_values('SHAP value', key=abs, ascending=False)
                    
                    # Criar gráfico de barras
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Definir cores com base no sinal do valor SHAP
                    colors = ['#2EB086' if x > 0 else '#FF4B4B' for x in shap_df['SHAP value']]
                    
                    # Mostrar apenas as top 10 características para clareza
                    top_n = min(10, len(shap_df))
                    
                    # Criar gráfico de barras horizontal
                    bars = ax.barh(
                        np.arange(top_n), 
                        shap_df['SHAP value'].head(top_n),
                        color=colors[:top_n],
                        alpha=0.8
                    )
                    
                    # Adicionar rótulos e título
                    ax.set_yticks(np.arange(top_n))
                    ax.set_yticklabels(shap_df['Feature'].head(top_n))
                    ax.set_title('Top 10 Variáveis que Impactam esta Decisão')
                    ax.set_xlabel('Impacto no Score (Valor SHAP)')
                    
                    # Adicionar linha vertical em x=0
                    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                    
                    # Adicionar valores como rótulos nas barras
                    for i, v in enumerate(shap_df['SHAP value'].head(top_n)):
                        ax.text(
                            v + (0.01 if v >= 0 else -0.01), 
                            i, 
                            f"{v:.3f}", 
                            va='center', 
                            ha='left' if v >= 0 else 'right'
                        )
                    
                    # Exibir gráfico
                    st.pyplot(fig)
                    
                    # Calcular contribuição total
                    positive_contribution = sum(v for v in shap_df['SHAP value'] if v > 0)
                    negative_contribution = sum(v for v in shap_df['SHAP value'] if v < 0)
                    net_contribution = positive_contribution + negative_contribution
                    
                    # Exibir análise de contribuição
                    st.write("### Análise de Contribuição")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fatores Positivos", f"{positive_contribution:.3f}")
                    with col2:
                        st.metric("Fatores Negativos", f"{negative_contribution:.3f}")
                    with col3:
                        st.metric("Contribuição Líquida", f"{net_contribution:.3f}")
                    
                    # Interpretação
                    st.write("""
                    ### Interpretação do Gráfico de Impacto
                    
                    Este gráfico mostra as principais variáveis que influenciaram a decisão do modelo para este cliente específico:
                    
                    - **Barras verdes**: Características que aumentam a probabilidade de ser classificado como bom pagador
                    - **Barras vermelhas**: Características que reduzem a probabilidade de ser classificado como bom pagador
                    - **Comprimento das barras**: Magnitude da influência de cada característica
                    
                    A diferença entre fatores positivos e negativos determina a classificação final do cliente.
                    """)
                    
                except Exception as e:
                    st.error(f"Erro ao processar valores SHAP para o exemplo: {e}")
                    
                    # Abordagem alternativa: mostrar importância com coeficientes, se disponível
                    st.info("Mostrando abordagem alternativa para explicação...")
                    
                    if hasattr(st.session_state.model, 'feature_importances_'):
                        # Para modelos baseados em árvores
                        importances = st.session_state.model.feature_importances_
                        feature_impact = example_data.copy()
                        feature_impact['Importância'] = [importances[feature_names.index(f)] 
                                                        if f in feature_names else 0 
                                                        for f in feature_impact['Característica']]
                        feature_impact['Impacto'] = feature_impact['Valor'] * feature_impact['Importância']
                        feature_impact = feature_impact.sort_values('Impacto', key=abs, ascending=False)
                        
                        st.dataframe(feature_impact)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_n = min(10, len(feature_impact))
                        colors = ['#2EB086' if x > 0 else '#FF4B4B' for x in feature_impact['Impacto'].head(top_n)]
                        
                        ax.barh(
                            np.arange(top_n),
                            feature_impact['Impacto'].head(top_n),
                            color=colors
                        )
                        
                        ax.set_yticks(np.arange(top_n))
                        ax.set_yticklabels(feature_impact['Característica'].head(top_n))
                        ax.set_title('Top 10 Variáveis com Maior Impacto (Análise Aproximada)')
                        ax.set_xlabel('Impacto Aproximado')
                        
                        st.pyplot(fig)
                    else:
                        st.warning("Não foi possível gerar uma explicação detalhada para este exemplo.")
                
            except Exception as e:
                st.error(f"Erro ao gerar explicação local: {e}")
                st.info("Tente selecionar outro exemplo ou utilizar outro tipo de visualização.")

        elif shap_plot_type == "Exemplo Individual":
            st.subheader("Análise Individual de Cliente")
            
            # Diagnóstico de SHAP para depuração
            st.write("### Diagnóstico SHAP")
            if st.session_state.shap_output_dim == 'multiclass':
                st.write(f"Tipo SHAP: multiclass, shape: {len(st.session_state.shap_values)}")
                st.write(f"Shape da classe positiva: {st.session_state.shap_values[st.session_state.shap_positive_class_idx].shape}")
            else:
                try:
                    st.write(f"Tipo SHAP: single, shape: {st.session_state.shap_values.shape}")
                except:
                    st.write("Tipo SHAP: single, sem shape determinado")
            
            st.write(f"Shape dos dados: {st.session_state.X_sample.shape}")
            
            # Permitir selecionar um cliente aleatório ou criar um novo
            analysis_option = st.radio(
                "Selecione como analisar o cliente:",
                ["Selecionar cliente existente", "Criar cliente hipotético"],
                horizontal=True
            )
            
            # Definir estatísticas dos dados para uso posterior (corrigindo X_mean undefined)
            X_mean = st.session_state.X_sample.mean()
            X_std = st.session_state.X_sample.std()
            X_min = st.session_state.X_sample.min()
            X_max = st.session_state.X_sample.max()
            
            if analysis_option == "Selecionar cliente existente":
                # Selecionar cliente para análise
                random_idx = st.slider("Selecione um ID de cliente:", 0, len(st.session_state.X_sample) - 1, 0)
                client_data = st.session_state.X_sample.iloc[random_idx:random_idx+1].copy()
                client_data_scaled = st.session_state.X_sample_scaled[random_idx:random_idx+1].copy()
                
                # Exibir dados do cliente
                st.subheader("Dados do Cliente")
                client_display = client_data.T.reset_index()
                client_display.columns = ['Característica', 'Valor']
                st.dataframe(client_display)
                
            else:  # Criar cliente hipotético
                st.subheader("Criar Cliente Hipotético")
                
                # Criar formulário para input de dados do cliente
                col1, col2 = st.columns(2)
                
                # Dicionário para armazenar os valores do cliente
                client_data_dict = {}
                
                with col1:
                    # Usar estatísticas para definir valores padrão e limites dos sliders
                    client_data_dict['idade'] = st.slider(
                        "Idade:", 
                        int(max(18, X_min['idade'])), 
                        int(min(80, X_max['idade'])),
                        int(X_mean['idade'])
                    )
                    
                    client_data_dict['renda_anual'] = st.slider(
                        "Renda Anual (R$):", 
                        int(X_min['renda_anual']), 
                        int(X_max['renda_anual']),
                        int(X_mean['renda_anual']),
                        5000
                    )
                    
                    client_data_dict['tempo_emprego'] = st.slider(
                        "Tempo de Emprego (anos):", 
                        int(X_min['tempo_emprego']), 
                        int(X_max['tempo_emprego']),
                        int(X_mean['tempo_emprego'])
                    )
                    
                    client_data_dict['tempo_residencia'] = st.slider(
                        "Tempo de Residência (anos):", 
                        int(X_min['tempo_residencia']), 
                        int(X_max['tempo_residencia']),
                        int(X_mean['tempo_residencia'])
                    )
                    
                    client_data_dict['qtd_contas'] = st.slider(
                        "Quantidade de Contas:", 
                        int(X_min['qtd_contas']), 
                        int(X_max['qtd_contas']),
                        int(X_mean['qtd_contas'])
                    )
                
                with col2:
                    client_data_dict['divida_total'] = st.slider(
                        "Dívida Total (R$):", 
                        int(X_min['divida_total']), 
                        int(X_max['divida_total']),
                        int(X_mean['divida_total']),
                        1000
                    )
                    
                    client_data_dict['taxa_utilizacao_credito'] = st.slider(
                        "Taxa de Utilização de Crédito:", 
                        float(X_min['taxa_utilizacao_credito']), 
                        float(X_max['taxa_utilizacao_credito']),
                        float(X_mean['taxa_utilizacao_credito']),
                        0.05
                    )
                    
                    client_data_dict['consultas_credito'] = st.slider(
                        "Consultas de Crédito (últimos 6 meses):", 
                        int(X_min['consultas_credito']), 
                        int(X_max['consultas_credito']),
                        int(X_mean['consultas_credito'])
                    )
                    
                    client_data_dict['atrasos_pagamento'] = st.slider(
                        "Atrasos de Pagamento (últimos 12 meses):", 
                        int(X_min['atrasos_pagamento']), 
                        int(X_max['atrasos_pagamento']),
                        int(X_mean['atrasos_pagamento'])
                    )
                    
                    client_data_dict['historico_pagamento'] = st.slider(
                        "Histórico de Pagamento (pontos):", 
                        int(X_min['historico_pagamento']), 
                        int(X_max['historico_pagamento']),
                        int(X_mean['historico_pagamento'])
                    )
                
                # Criar DataFrame com dados do cliente
                client_data = pd.DataFrame([client_data_dict])
                
                # Normalizar dados usando o mesmo scaler do modelo
                try:
                    client_data_scaled = st.session_state.scaler.transform(client_data)
                except Exception as e:
                    st.error(f"Erro ao normalizar dados do cliente: {e}")
                    client_data_scaled = client_data.values  # Usar valores não normalizados como fallback
            
            # Divider
            st.markdown("---")
            
            # Predizer score de crédito
            try:
                credit_proba = st.session_state.model.predict_proba(client_data_scaled)[0, 1]
                credit_score = credit_proba * 1000  # Escala de 0 a 1000
            
                # Determinar categoria de risco
                risk_category = pd.cut(
                    [credit_score], 
                    bins=[0, 300, 500, 700, 850, 1000], 
                    labels=['Muito Alto Risco', 'Alto Risco', 'Médio Risco', 'Baixo Risco', 'Muito Baixo Risco']
                )[0]
                
                # Exibir score e categoria
                st.subheader("Análise de Crédito")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Visualizar score como um gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = credit_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Credit Score"},
                        gauge = {
                            'axis': {'range': [0, 1000], 'tickwidth': 1},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 300], 'color': "#FF4B4B"},
                                {'range': [300, 500], 'color': "#FFA500"},
                                {'range': [500, 700], 'color': "#FFFF00"},
                                {'range': [700, 850], 'color': "#90EE90"},
                                {'range': [850, 1000], 'color': "#2EB086"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 500
                            }
                        }
                    ))
                    
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Categoria de Risco", f"{risk_category}")
                    
                    # Decisão de aprovação
                    approval_decision = "Aprovado" if credit_score > 500 else "Reprovado"
                    approval_color = "#2EB086" if approval_decision == "Aprovado" else "#FF4B4B"
                    
                    st.markdown(f"### Decisão: <span style='color:{approval_color}'>{approval_decision}</span>", unsafe_allow_html=True)
                    
                    # Taxa de juros estimada
                    if approval_decision == "Aprovado":
                        if credit_score > 850:
                            juros = "8.9%"
                        elif credit_score > 700:
                            juros = "12.5%"
                        elif credit_score > 600:
                            juros = "18.0%"
                        else:
                            juros = "24.9%"
                            
                        st.metric("Taxa de Juros Estimada", juros)
                
                with col3:
                    # Probabilidade de inadimplência
                    default_prob = 1 - credit_proba
                    
                    # Determinar nível de risco
                    if default_prob < 0.1:
                        risk_level = "Baixo"
                        risk_color = "#2EB086"
                    elif default_prob < 0.3:
                        risk_level = "Moderado"
                        risk_color = "#FFA500"
                    else:
                        risk_level = "Alto"
                        risk_color = "#FF4B4B"
                    
                    st.metric("Probabilidade de Inadimplência", f"{default_prob:.1%}")
                    st.markdown(f"### Risco: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                
                # Divider
                st.markdown("---")
                
                # Análise SHAP
                st.subheader("Explicação do Score de Crédito")
                st.write("Este gráfico mostra como cada característica contribuiu para o score final deste cliente.")
                
                try:
                    # Abordagem simplificada para explicação usando importâncias do modelo
                    if hasattr(st.session_state.model, 'feature_importances_'):
                        # Para modelos baseados em árvores (como RandomForest, XGBoost)
                        importances = st.session_state.model.feature_importances_
                        
                        # Calcular direção do impacto baseado nos valores do cliente vs média
                        client_features = client_data.iloc[0].values
                        mean_features = X_mean.values
                        
                        # Criar valores SHAP aproximados
                        # Positivo se maior que média e feature é positiva (ou menor que média e feature é negativa)
                        # Negativo caso contrário
                        impact_signs = []
                        for i, (client_val, mean_val) in enumerate(zip(client_features, mean_features)):
                            # Determinar importância pela característica
                            feature_importance = importances[i]
                            
                            # Calcular diferença normalizada
                            diff = (client_val - mean_val) / (X_std.values[i] if X_std.values[i] > 0 else 1)
                            
                            # Determinar sinal baseado na importância da característica e seu valor
                            # Esta é uma aproximação simples que pode ser refinada
                            if feature_importance == 0:
                                impact_signs.append(0)
                            else:
                                # Diferença normalizada x importância
                                # Mais intensa para valores mais distantes da média
                                impact_signs.append(diff * feature_importance * 5)  # Multiplicador para aumentar escala
                        
                        # Criar DataFrame para visualização
                        client_shap_df = pd.DataFrame({
                            'Feature': client_data.columns,
                            'SHAP value': impact_signs,
                            'Feature value': client_features
                        }).sort_values('SHAP value', key=abs, ascending=False)
                        
                        # Criar gráfico de barras
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Definir cores com base no sinal do valor SHAP
                        colors = ['#2EB086' if x > 0 else '#FF4B4B' for x in client_shap_df['SHAP value']]
                        
                        # Criar gráfico de barras
                        sns.barplot(
                            x='SHAP value',
                            y='Feature',
                            data=client_shap_df,
                            palette=colors,
                            ax=ax
                        )
                        
                        # Definir título e rótulos
                        ax.set_title('Fatores que Impactam o Score do Cliente')
                        ax.set_xlabel('Impacto no Score (Valor aproximado)')
                        ax.set_ylabel('Característica')
                        
                        # Adicionar linha vertical em x=0
                        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                        
                        # Adicionar os valores das características ao lado das barras
                        for i, (_, row) in enumerate(client_shap_df.iterrows()):
                            feature_name = row['Feature']
                            feature_val = row['Feature value']
                            
                            # Formatação especial para certas características
                            if feature_name == 'taxa_utilizacao_credito':
                                display_val = f"{feature_val*100:.1f}%"
                            elif feature_name in ['renda_anual', 'divida_total']:
                                display_val = f"R$ {feature_val:,.2f}"
                            else:
                                display_val = f"{feature_val:.2f}"
                                
                            ax.text(
                                row['SHAP value'] + (0.05 if row['SHAP value'] >= 0 else -0.05), 
                                i, 
                                f"Valor: {display_val}", 
                                va='center', 
                                ha='left' if row['SHAP value'] >= 0 else 'right',
                                fontsize=9,
                                alpha=0.7
                            )
                        
                        # Exibir gráfico
                        st.pyplot(fig)
                        
                        # Recomendações
                        st.subheader("Recomendações para Melhorar o Score")
                        
                        # Identificar características negativas mais impactantes
                        negative_impacts = client_shap_df[client_shap_df['SHAP value'] < 0].head(3)
                        
                        if len(negative_impacts) > 0:
                            st.write("Para melhorar seu score de crédito, considere trabalhar nestes aspectos:")
                            
                            for _, row in negative_impacts.iterrows():
                                feature = row['Feature']
                                value = row['Feature value']
                                
                                if feature == 'atrasos_pagamento':
                                    st.markdown(f"- **Reduzir Atrasos de Pagamento**: Atualmente com {value:.0f} atrasos. Pague suas contas em dia e evite novos atrasos.")
                                elif feature == 'consultas_credito':
                                    st.markdown(f"- **Reduzir Consultas de Crédito**: Atualmente com {value:.0f} consultas. Evite solicitar muitos créditos em curto período.")
                                elif feature == 'taxa_utilizacao_credito':
                                    st.markdown(f"- **Reduzir Taxa de Utilização de Crédito**: Atualmente em {value*100:.0f}%. Tente utilizar menos do seu limite disponível.")
                                elif feature == 'divida_total':
                                    st.markdown(f"- **Reduzir Dívida Total**: Atualmente R$ {value:.2f}. Trabalhe para diminuir seu endividamento total.")
                                elif feature == 'qtd_contas':
                                    st.markdown(f"- **Reduzir Número de Contas**: Atualmente com {value:.0f} contas. Considere consolidar ou fechar algumas contas.")
                                elif feature == 'historico_pagamento':
                                    st.markdown(f"- **Melhorar Histórico de Pagamento**: Atual pontuação {value:.0f}. Mantenha pagamentos em dia para aumentar esta pontuação.")
                                else:
                                    st.markdown(f"- **Melhorar {feature}**: Valor atual {value:.2f}. Este fator está impactando negativamente seu score.")
                        else:
                            st.write("Seu perfil já apresenta boas características de crédito!")
                    else:
                        st.warning("O tipo de modelo atual não suporta importância de características direta. Não foi possível gerar a explicação SHAP detalhada.")
                
                except Exception as e:
                    st.error(f"Erro ao calcular valores SHAP para este cliente: {e}")
                    
                    # Oferecer uma visualização alternativa
                    st.info("Mostrando informações gerais do cliente sem análise SHAP detalhada.")
                    
                    # Destacar fatores de risco baseados em conhecimento geral de crédito
                    st.write("### Fatores de Risco Potenciais:")
                    risk_factors = []
                    
                    # Verificar fatores comuns de risco
                    if 'atrasos_pagamento' in client_data and client_data['atrasos_pagamento'].values[0] > 0:
                        risk_factors.append(f"- **Atrasos de Pagamento**: {client_data['atrasos_pagamento'].values[0]} atrasos")
                        
                    if 'taxa_utilizacao_credito' in client_data and client_data['taxa_utilizacao_credito'].values[0] > 0.7:
                        taxa = client_data['taxa_utilizacao_credito'].values[0] * 100
                        risk_factors.append(f"- **Alta Taxa de Utilização de Crédito**: {taxa:.0f}%")
                        
                    if 'consultas_credito' in client_data and client_data['consultas_credito'].values[0] > 3:
                        risk_factors.append(f"- **Muitas Consultas de Crédito**: {client_data['consultas_credito'].values[0]} consultas")
                    
                    if 'historico_pagamento' in client_data and client_data['historico_pagamento'].values[0] < 60:
                        risk_factors.append(f"- **Baixo Histórico de Pagamento**: {client_data['historico_pagamento'].values[0]} pontos")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.markdown(factor)
                    else:
                        st.write("Não foram identificados fatores de risco significativos.")
                    
                    # Criar uma análise de impacto aproximada baseada em regras gerais de crédito
                    st.subheader("Análise de Impacto Aproximada")
                    
                    impact_dict = {
                        'idade': 0.05,
                        'renda_anual': 0.15,
                        'tempo_emprego': 0.10,
                        'tempo_residencia': 0.05,
                        'qtd_contas': -0.05,
                        'divida_total': -0.10,
                        'taxa_utilizacao_credito': -0.20,
                        'consultas_credito': -0.10,
                        'atrasos_pagamento': -0.20,
                        'historico_pagamento': 0.15
                    }
                    
                    # Calcular impacto aproximado
                    impacts = []
                    for col in client_data.columns:
                        if col in impact_dict:
                            # Normalizar o valor em relação à média
                            mean_val = X_mean[col]
                            std_val = max(0.0001, X_std[col])  # Evitar divisão por zero
                            
                            normalized_val = (client_data[col].values[0] - mean_val) / std_val
                            impact = normalized_val * impact_dict[col]
                            
                            impacts.append({
                                'Feature': col,
                                'Impact': impact,
                                'Value': client_data[col].values[0]
                            })
                    
                    # Criar DataFrame e ordenar
                    impact_df = pd.DataFrame(impacts).sort_values('Impact', key=abs, ascending=False)
                    
                    # Criar gráfico de barras
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#2EB086' if x > 0 else '#FF4B4B' for x in impact_df['Impact']]
                    
                    sns.barplot(
                        x='Impact', 
                        y='Feature',
                        data=impact_df,
                        palette=colors,
                        ax=ax
                    )
                    
                    ax.set_title('Impacto Aproximado no Score (Baseado em Regras Gerais)')
                    ax.set_xlabel('Impacto Estimado')
                    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
                    
                    st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Erro ao calcular score de crédito: {e}")
                st.warning("Não foi possível calcular o score de crédito para este cliente.")

# Simulador de Crédito
elif app_mode == "Simulador de Crédito":
    st.header("Simulador de Decisões de Crédito")
    
    st.write("""
    Este simulador permite testar diferentes cenários de aprovação de crédito, 
    definindo políticas de risco e analisando o impacto nas taxas de aprovação e inadimplência.
    """)
    
    # Definição das políticas de risco
    st.subheader("Definição de Política de Risco")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score mínimo para aprovação
        min_score = st.slider(
            "Score Mínimo para Aprovação (0-1000):", 
            min_value=0, 
            max_value=1000, 
            value=500, 
            step=10,
            help="Clientes com score abaixo deste valor serão rejeitados"
        )
        
        # Taxa de juros base
        base_interest_rate = st.slider(
            "Taxa de Juros Base (% a.a.):", 
            min_value=5.0, 
            max_value=50.0, 
            value=15.0, 
            step=0.5,
            help="Taxa de juros base anual para clientes de baixo risco"
        )
        
        # Valor máximo do empréstimo (% da renda)
        max_loan_income_ratio = st.slider(
            "Valor Máximo do Empréstimo (múltiplo da renda):", 
            min_value=0.5, 
            max_value=10.0, 
            value=5.0, 
            step=0.5,
            help="Limite máximo do empréstimo como múltiplo da renda anual"
        )
    
    with col2:
        # Fatores de ajuste de taxa
        st.write("**Ajustes de Taxa de Juros por Categoria de Risco:**")
        
        risk_adjustments = {}
        risk_adjustments['Muito Alto Risco'] = st.slider("Adicional para Muito Alto Risco (%):", 0.0, 30.0, 25.0, 0.5)
        risk_adjustments['Alto Risco'] = st.slider("Adicional para Alto Risco (%):", 0.0, 20.0, 15.0, 0.5)
        risk_adjustments['Médio Risco'] = st.slider("Adicional para Médio Risco (%):", 0.0, 10.0, 5.0, 0.5)
        risk_adjustments['Baixo Risco'] = st.slider("Adicional para Baixo Risco (%):", 0.0, 5.0, 0.0, 0.5)
        risk_adjustments['Muito Baixo Risco'] = st.slider("Desconto para Muito Baixo Risco (%):", 0.0, 5.0, 2.0, 0.5)
    
    # Regras adicionais
    st.subheader("Regras Adicionais")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_debt_income_ratio = st.slider(
            "Dívida Máxima (% da renda):", 
            min_value=10, 
            max_value=100, 
            value=50, 
            step=5,
            help="Dívida total máxima como percentual da renda"
        )
    
    with col2:
        min_employment_time = st.slider(
            "Tempo Mínimo de Emprego (anos):", 
            min_value=0, 
            max_value=5, 
            value=1, 
            step=1,
            help="Tempo mínimo de emprego para aprovação"
        )
    
    with col3:
        max_late_payments = st.slider(
            "Máximo de Atrasos Permitidos:", 
            min_value=0, 
            max_value=12, 
            value=2, 
            step=1,
            help="Número máximo de atrasos de pagamento permitidos"
        )
    
    # Botão para simular
    simulate_button = st.button("Simular Política de Crédito", type="primary")
    
    if simulate_button:
        with st.spinner("Simulando política de crédito..."):
            # Calcular scores para todos os clientes
            X_all = st.session_state.data.drop('bom_pagador', axis=1)
            X_all_scaled = st.session_state.scaler.transform(X_all)
            all_scores = st.session_state.model.predict_proba(X_all_scaled)[:, 1] * 1000
            
            # Criar DataFrame com resultados
            simulation_results = pd.DataFrame({
                'score': all_scores,
                'bom_pagador': st.session_state.data['bom_pagador'],
                'idade': st.session_state.data['idade'],
                'renda_anual': st.session_state.data['renda_anual'],
                'tempo_emprego': st.session_state.data['tempo_emprego'],
                'divida_total': st.session_state.data['divida_total'],
                'atrasos_pagamento': st.session_state.data['atrasos_pagamento']
            })
            
            # Categorizar scores
            simulation_results['categoria_risco'] = pd.cut(
                simulation_results['score'], 
                bins=[0, 300, 500, 700, 850, 1000], 
                labels=['Muito Alto Risco', 'Alto Risco', 'Médio Risco', 'Baixo Risco', 'Muito Baixo Risco']
            )
            
            # Aplicar critérios de aprovação
            simulation_results['divida_renda_ratio'] = simulation_results['divida_total'] / simulation_results['renda_anual'] * 100
            
            # Verificar aprovação pelo score
            simulation_results['aprovado_score'] = simulation_results['score'] >= min_score
            
            # Verificar aprovação por tempo de emprego
            simulation_results['aprovado_emprego'] = simulation_results['tempo_emprego'] >= min_employment_time
            
            # Verificar aprovação por atrasos
            simulation_results['aprovado_atrasos'] = simulation_results['atrasos_pagamento'] <= max_late_payments
            
            # Verificar aprovação por razão dívida/renda
            simulation_results['aprovado_divida'] = simulation_results['divida_renda_ratio'] <= max_debt_income_ratio
            
            # Decisão final de aprovação (todas as regras devem ser satisfeitas)
            simulation_results['aprovado'] = (
                simulation_results['aprovado_score'] & 
                simulation_results['aprovado_emprego'] & 
                simulation_results['aprovado_atrasos'] & 
                simulation_results['aprovado_divida']
            )
            
            # Calcular valor máximo do empréstimo
            simulation_results['valor_max_emprestimo'] = simulation_results['renda_anual'] * max_loan_income_ratio
            
            # Calcular taxa de juros personalizada
            simulation_results['taxa_juros'] = base_interest_rate
            
            # Ajustar taxa por categoria de risco
            for categoria, ajuste in risk_adjustments.items():
                # Para Muito Baixo Risco, aplicamos desconto
                if categoria == 'Muito Baixo Risco':
                    simulation_results.loc[simulation_results['categoria_risco'] == categoria, 'taxa_juros'] -= ajuste
                else:
                    simulation_results.loc[simulation_results['categoria_risco'] == categoria, 'taxa_juros'] += ajuste
            
            # Limitar taxa mínima a 1%
            simulation_results['taxa_juros'] = simulation_results['taxa_juros'].clip(lower=1.0)
            
            # Resultados da simulação
            st.subheader("Resultados da Simulação")
            
            # Métricas principais
            total_clients = len(simulation_results)
            approved_clients = simulation_results['aprovado'].sum()
            approval_rate = approved_clients / total_clients
            
            # Calcular taxa de inadimplência esperada na carteira aprovada
            if approved_clients > 0:
                expected_default_rate = 1 - simulation_results[simulation_results['aprovado']]['bom_pagador'].mean()
            else:
                expected_default_rate = 0
            
            # Calcular valor total dos empréstimos aprovados
            total_loan_value = simulation_results[simulation_results['aprovado']]['valor_max_emprestimo'].sum()
            
            # Exibir métricas em cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Taxa de Aprovação", f"{approval_rate:.2%}")
                st.metric("Clientes Aprovados", f"{approved_clients} de {total_clients}")
            
            with col2:
                st.metric("Taxa de Inadimplência Estimada", f"{expected_default_rate:.2%}")
                st.metric("Valor Total da Carteira", f"R$ {total_loan_value:,.2f}")
            
            with col3:
                st.metric("Taxa de Juros Média", f"{simulation_results[simulation_results['aprovado']]['taxa_juros'].mean():.2f}% a.a.")
                st.metric("Score Médio dos Aprovados", f"{simulation_results[simulation_results['aprovado']]['score'].mean():.0f}")
            
            # Gráfico de aprovação por categoria de risco
            st.subheader("Aprovação por Categoria de Risco")
            
            approval_by_category = simulation_results.groupby('categoria_risco')['aprovado'].mean().reset_index()
            approval_by_category.columns = ['Categoria de Risco', 'Taxa de Aprovação']
            approval_by_category['Taxa de Aprovação'] = approval_by_category['Taxa de Aprovação'] * 100
            
            fig = px.bar(
                approval_by_category,
                x='Categoria de Risco',
                y='Taxa de Aprovação',
                color='Taxa de Aprovação',
                labels={'Taxa de Aprovação': 'Taxa de Aprovação (%)'},
                title='Taxa de Aprovação por Categoria de Risco',
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': ['Muito Alto Risco', 'Alto Risco', 'Médio Risco', 'Baixo Risco', 'Muito Baixo Risco']})
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribuição das taxas de juros
            st.subheader("Distribuição das Taxas de Juros (Aprovados)")
            
            if approved_clients > 0:
                fig = px.histogram(
                    simulation_results[simulation_results['aprovado']], 
                    x='taxa_juros',
                    nbins=20,
                    labels={'taxa_juros': 'Taxa de Juros (% a.a.)'},
                    title='Distribuição das Taxas de Juros para Clientes Aprovados',
                    color_discrete_sequence=['#3366CC']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Nenhum cliente aprovado com os critérios atuais.")
            
            # Motivos de rejeição
            st.subheader("Motivos de Rejeição")
            
            rejected_reasons = {
                'Score Baixo': (~simulation_results['aprovado_score']).sum(),
                'Tempo de Emprego Insuficiente': (~simulation_results['aprovado_emprego']).sum(),
                'Muitos Atrasos': (~simulation_results['aprovado_atrasos']).sum(),
                'Dívida/Renda Alta': (~simulation_results['aprovado_divida']).sum()
            }
            
            reasons_df = pd.DataFrame({
                'Motivo': list(rejected_reasons.keys()),
                'Quantidade': list(rejected_reasons.values())
            })
            
            fig = px.bar(
                reasons_df,
                x='Motivo',
                y='Quantidade',
                color='Quantidade',
                title='Motivos de Rejeição',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Valor esperado da carteira
            st.subheader("Análise Financeira da Carteira")
            
            # Estimativa simplificada de lucro
            if approved_clients > 0:
                # Assumindo empréstimo médio de 12 meses
                avg_loan_approved = simulation_results[simulation_results['aprovado']]['valor_max_emprestimo'].mean()
                avg_interest_rate = simulation_results[simulation_results['aprovado']]['taxa_juros'].mean() / 100
                
                # Receita esperada de juros (simplificada)
                expected_interest_revenue = total_loan_value * avg_interest_rate
                
                # Perda esperada devido à inadimplência (simplificada)
                expected_loss = total_loan_value * expected_default_rate
                
                # Lucro esperado
                expected_profit = expected_interest_revenue - expected_loss
                
                # ROI estimado
                roi = expected_profit / total_loan_value * 100 if total_loan_value > 0 else 0
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Receita Esperada de Juros", f"R$ {expected_interest_revenue:,.2f}")
                    st.metric("Perda Esperada (Inadimplência)", f"R$ {expected_loss:,.2f}")
                
                with col2:
                    st.metric("Lucro Esperado", f"R$ {expected_profit:,.2f}")
                    st.metric("ROI Estimado", f"{roi:.2f}%")
                
                # Tabela com análise detalhada por categoria de risco
                st.subheader("Análise por Categoria de Risco")
                
                # Agrupar por categoria
                category_analysis = simulation_results[simulation_results['aprovado']].groupby('categoria_risco').agg({
                    'bom_pagador': 'mean',
                    'valor_max_emprestimo': 'sum',
                    'taxa_juros': 'mean',
                    'aprovado': 'count'
                }).reset_index()
                
                # Renomear colunas
                category_analysis.columns = [
                    'Categoria de Risco', 
                    'Taxa de Bons Pagadores', 
                    'Valor Total', 
                    'Taxa de Juros Média',
                    'Quantidade de Clientes'
                ]
                
                # Calcular inadimplência
                category_analysis['Taxa de Inadimplência'] = 1 - category_analysis['Taxa de Bons Pagadores']
                
                # Calcular receita e perda
                category_analysis['Receita de Juros'] = category_analysis['Valor Total'] * (category_analysis['Taxa de Juros Média'] / 100)
                category_analysis['Perda por Inadimplência'] = category_analysis['Valor Total'] * category_analysis['Taxa de Inadimplência']
                category_analysis['Lucro Estimado'] = category_analysis['Receita de Juros'] - category_analysis['Perda por Inadimplência']
                
                # Formatar valores
                category_analysis['Taxa de Bons Pagadores'] = category_analysis['Taxa de Bons Pagadores'].apply(lambda x: f"{x:.2%}")
                category_analysis['Taxa de Inadimplência'] = category_analysis['Taxa de Inadimplência'].apply(lambda x: f"{x:.2%}")
                category_analysis['Taxa de Juros Média'] = category_analysis['Taxa de Juros Média'].apply(lambda x: f"{x:.2f}%")
                category_analysis['Valor Total'] = category_analysis['Valor Total'].apply(lambda x: f"R$ {x:,.2f}")
                category_analysis['Receita de Juros'] = category_analysis['Receita de Juros'].apply(lambda x: f"R$ {x:,.2f}")
                category_analysis['Perda por Inadimplência'] = category_analysis['Perda por Inadimplência'].apply(lambda x: f"R$ {x:,.2f}")
                category_analysis['Lucro Estimado'] = category_analysis['Lucro Estimado'].apply(lambda x: f"R$ {x:,.2f}")
                
                st.dataframe(category_analysis)
            else:
                st.warning("Nenhum cliente aprovado para análise financeira.")
    
    # Dicas de otimização
    with st.expander("Dicas para Otimização de Políticas"):
        st.markdown("""
        ### Como otimizar sua política de crédito
        
        1. **Balanceamento de risco e retorno**:
           - Políticas muito restritivas (score mínimo alto) reduzem a inadimplência, mas também limitam o crescimento.
           - Políticas muito permissivas aumentam o volume de empréstimos, mas com maior risco.
        
        2. **Segmentação de clientes**:
           - Crie políticas diferentes para segmentos distintos (por idade, renda, etc.).
           - Clientes de baixo risco podem receber limites maiores e taxas menores.
        
        3. **Ajuste dinâmico**:
           - Monitore o desempenho da carteira e ajuste os parâmetros periodicamente.
           - Considere fatores macroeconômicos na definição de políticas.
        
        4. **Precificação baseada em risco**:
           - A taxa de juros deve compensar o risco estimado.
           - Calcule o ponto de equilíbrio entre atratividade da taxa e compensação do risco.
        """)

# Comparação de Modelos
elif app_mode == "Comparação de Modelos":
    st.header("Comparação de Modelos de Machine Learning")
    
    st.write("""
    Esta seção permite comparar diferentes algoritmos de machine learning para credit scoring.
    Compare métricas de performance, curvas ROC, e distribuições de scores para escolher o modelo ideal.
    """)
    
    # Opções de configuração
    st.subheader("Configuração da Análise")
    col1, col2 = st.columns(2)
    
    with col1:
        validation_method = st.radio(
            "Método de Validação:",
            ["Holdout (Train/Test Split)", "Validação Cruzada (5-fold)"],
            horizontal=True
        )
    
    with col2:
        selected_models = st.multiselect(
            "Selecione os Modelos para Comparação:",
            ["Random Forest", "XGBoost", "LightGBM", "Rede Neural (MLP)", "Regressão Logística"],
            default=["Random Forest", "XGBoost", "Regressão Logística"]
        )
    
    if not selected_models:
        st.warning("Por favor, selecione pelo menos um modelo para análise.")
    else:
        # Mapear seleções para instâncias de modelos
        model_mapping = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(objective='binary:logistic', random_state=42),
            "LightGBM": LGBMClassifier(random_state=42),
            "Rede Neural (MLP)": MLPClassifier(max_iter=300, random_state=42),
            "Regressão Logística": LogisticRegression(random_state=42)
        }
        
        models = [model_mapping[name] for name in selected_models]
        
        # Botão para iniciar a comparação
        if st.button("Comparar Modelos Selecionados", type="primary"):
            # Divisão entre features e target
            X = st.session_state.data.drop('bom_pagador', axis=1)
            y = st.session_state.data['bom_pagador']
            
            # Normalização dos dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Lista para armazenar resultados
            results = []
            roc_curves = []
            pr_curves = []
            score_distributions = []
            
            # Divisão entre treino e teste (para método holdout)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Preparar figura para curvas ROC
            fig_roc = go.Figure()
            fig_roc.update_layout(
                title='Curvas ROC',
                xaxis_title='Taxa de Falsos Positivos',
                yaxis_title='Taxa de Verdadeiros Positivos',
                legend=dict(x=0.1, y=0, orientation='h')
            )
            
            # Preparar figura para curvas Precision-Recall
            fig_pr = go.Figure()
            fig_pr.update_layout(
                title='Curvas Precision-Recall',
                xaxis_title='Recall',
                yaxis_title='Precision',
                legend=dict(x=0.1, y=0, orientation='h')
            )
            
            # Analisar cada modelo
            for model in models:
                model_name = model.__class__.__name__
                with st.spinner(f"Analisando {model_name}..."):
                    
                    if validation_method == "Holdout (Train/Test Split)":
                        # Treinar no conjunto de treino
                        model.fit(X_train, y_train)
                        
                        # Prever no conjunto de teste
                        y_pred = model.predict(X_test)
                        y_prob = model.predict_proba(X_test)[:, 1]
                        
                        # Calcular métricas
                        report = classification_report(y_test, y_pred, output_dict=True)
                        
                        # Calcular curva ROC
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        
                        # Adicionar à figura ROC
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, 
                            name=f"{model_name} (AUC={roc_auc:.3f})",
                            mode='lines'
                        ))
                        
                        # Calcular curva Precision-Recall
                        precision, recall, _ = precision_recall_curve(y_test, y_prob)
                        
                        # Adicionar à figura Precision-Recall
                        fig_pr.add_trace(go.Scatter(
                            x=recall, y=precision,
                            name=f"{model_name}",
                            mode='lines'
                        ))
                        
                        # Calcular scores
                        scores = y_prob * 1000
                    
                    else:  # Validação Cruzada
                        # Calcular métricas via validação cruzada
                        cv_accuracy = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
                        cv_precision = cross_val_score(model, X_scaled, y, cv=5, scoring='precision')
                        cv_recall = cross_val_score(model, X_scaled, y, cv=5, scoring='recall')
                        cv_f1 = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
                        
                        # Treinar no conjunto completo para ROC e distribuições
                        model.fit(X_scaled, y)
                        y_prob = model.predict_proba(X_scaled)[:, 1]
                        
                        # Calcular ROC via validação cruzada (aproximação)
                        fpr, tpr, _ = roc_curve(y, y_prob)
                        roc_auc = auc(fpr, tpr)
                        
                        # Adicionar à figura ROC
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, 
                            name=f"{model_name} (AUC={roc_auc:.3f})",
                            mode='lines'
                        ))
                        
                        # Calcular curva Precision-Recall
                        precision, recall, _ = precision_recall_curve(y, y_prob)
                        
                        # Adicionar à figura Precision-Recall
                        fig_pr.add_trace(go.Scatter(
                            x=recall, y=precision,
                            name=f"{model_name}",
                            mode='lines'
                        ))
                        
                        # Calcular scores
                        scores = y_prob * 1000
                        
                        # Criar relatório similar ao do método holdout
                        report = {
                            'accuracy': np.mean(cv_accuracy),
                            '1': {
                                'precision': np.mean(cv_precision),
                                'recall': np.mean(cv_recall),
                                'f1-score': np.mean(cv_f1)
                            }
                        }
                    
                    # Armazenar resultados
                    results.append({
                        'Modelo': model_name,
                        'Acurácia': report['accuracy'],
                        'Precisão': report['1']['precision'],
                        'Recall': report['1']['recall'],
                        'F1-Score': report['1']['f1-score'],
                        'AUC-ROC': roc_auc
                    })
                    
                    # Armazenar distribuição de scores
                    score_distributions.append({
                        'modelo': model_name,
                        'scores': scores
                    })
            
            # Adicionar linha de referência à curva ROC
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Chance',
                mode='lines',
                line=dict(dash='dash', color='gray')
            ))
            
            # Criar DataFrame com resultados
            results_df = pd.DataFrame(results)
            
            # Exibir tabela de resultados
            st.subheader("Comparação de Métricas")
            
            # Formatar para exibição em porcentagem
            display_df = results_df.copy()
            for col in ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']:
                display_df[col] = display_df[col].map(lambda x: f"{x:.2%}")
                
            st.dataframe(display_df)
            
            # Exibir gráfico de barras comparativo
            st.subheader("Comparação Visual de Métricas")
            
            metrics_df = pd.melt(
                results_df, 
                id_vars=['Modelo'], 
                value_vars=['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC'],
                var_name='Métrica', 
                value_name='Valor'
            )
            
            fig_bar = px.bar(
                metrics_df, 
                x='Métrica', 
                y='Valor',
                color='Modelo',
                barmode='group',
                text_auto='.0%'
            )
            
            fig_bar.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Exibir curvas ROC
            st.subheader("Curvas ROC")
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Exibir curvas Precision-Recall
            st.subheader("Curvas Precision-Recall")
            st.plotly_chart(fig_pr, use_container_width=True)
            
            # Exibir distribuições de scores
            st.subheader("Comparação das Distribuições de Scores")
            
            # Criar DataFrame para histograma
            scores_data = []
            for dist in score_distributions:
                for score in dist['scores']:
                    scores_data.append({
                        'Modelo': dist['modelo'],
                        'Score': score
                    })
            
            scores_df = pd.DataFrame(scores_data)
            
            fig_hist = px.histogram(
                scores_df,
                x='Score',
                color='Modelo',
                barmode='overlay',
                opacity=0.7,
                nbins=50,
                range_x=[0, 1000]
            )
            
            fig_hist.update_layout(
                title='Distribuição dos Scores por Modelo',
                xaxis_title='Credit Score',
                yaxis_title='Frequência'
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Insights e recomendações
            st.subheader("Insights e Recomendações")
            
            # Encontrar melhor modelo para cada métrica
            best_accuracy = results_df.loc[results_df['Acurácia'].idxmax()]
            best_precision = results_df.loc[results_df['Precisão'].idxmax()]
            best_recall = results_df.loc[results_df['Recall'].idxmax()]
            best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]
            best_auc = results_df.loc[results_df['AUC-ROC'].idxmax()]
            
            st.write(f"""
            ### Análise dos Resultados
            
            - **Melhor Acurácia**: {best_accuracy['Modelo']} ({best_accuracy['Acurácia']:.2%})
            - **Melhor Precisão**: {best_precision['Modelo']} ({best_precision['Precisão']:.2%})
            - **Melhor Recall**: {best_recall['Modelo']} ({best_recall['Recall']:.2%})
            - **Melhor F1-Score**: {best_f1['Modelo']} ({best_f1['F1-Score']:.2%})
            - **Melhor AUC-ROC**: {best_auc['Modelo']} ({best_auc['AUC-ROC']:.2%})
            
            #### Qual modelo escolher?
            
            - Para **minimizar falsos positivos** (evitar aprovar clientes ruins), priorize a **Precisão** ({best_precision['Modelo']}).
            - Para **minimizar falsos negativos** (evitar rejeitar bons clientes), priorize o **Recall** ({best_recall['Modelo']}).
            - Para um **equilíbrio geral**, considere o **F1-Score** ou **AUC-ROC** ({best_f1['Modelo']} ou {best_auc['Modelo']}).
            
            #### Próximos passos recomendados:
            
            1. Otimize hiperparâmetros do modelo escolhido
            2. Considere ensemble de modelos para melhorar resultados
            3. Implemente validação específica no contexto de negócio
            """)
            
            # Opção para salvar o melhor modelo
            st.subheader("Salvar Modelo")
            save_model_option = st.radio(
                "Escolha o modelo para usar no sistema:",
                [m['Modelo'] for m in results],
                horizontal=True
            )
            
            if st.button("Usar este modelo como padrão"):
                # Identificar o modelo escolhido
                selected_idx = next(i for i, m in enumerate(results) if m['Modelo'] == save_model_option)
                selected_model = models[selected_idx]
                
                # Treinar no conjunto completo
                X = st.session_state.data.drop('bom_pagador', axis=1)
                y = st.session_state.data['bom_pagador']
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                selected_model.fit(X_scaled, y)
                
                # Salvar no session state
                st.session_state.model = selected_model
                st.session_state.scaler = scaler
                
                st.success(f"Modelo {save_model_option} definido como padrão do sistema!")

# Análise de Cobrança
elif app_mode == "Análise de Cobrança":
    st.header("Análise e Estratégia de Cobrança")
    
    st.write("""
    Esta seção permite analisar diferentes estratégias de cobrança e prever
    a probabilidade de recuperação de crédito com base em características do cliente
    e abordagens de negociação.
    """)
    
    # Verificar se já existem dados de cobrança ou gerar novos
    if 'collection_data' not in st.session_state:
        with st.spinner("Gerando dados sintéticos de cobrança..."):
            st.session_state.collection_data = generate_collection_data(st.session_state.data)
    
    # Tabs para organizar a interface
    tab1, tab2, tab3 = st.tabs(["Visão Geral", "Análise de Estratégias", "Simulador de Recuperação"])

    with tab1:
        st.subheader("Visão Geral da Carteira em Cobrança")
        
        # Estatísticas gerais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_valor = st.session_state.collection_data['valor_atraso'].sum()
            st.metric("Valor Total em Atraso", f"R$ {total_valor:,.2f}")
            
            qtd_devedores = len(st.session_state.collection_data)
            st.metric("Quantidade de Devedores", f"{qtd_devedores}")
        
        with col2:
            valor_medio = st.session_state.collection_data['valor_atraso'].mean()
            st.metric("Valor Médio em Atraso", f"R$ {valor_medio:,.2f}")
            
            dias_medio = st.session_state.collection_data['dias_atraso'].mean()
            st.metric("Dias de Atraso (Média)", f"{dias_medio:.0f} dias")
        
        with col3:
            taxa_recuperacao = st.session_state.collection_data['sucesso_cobranca'].mean()
            st.metric("Taxa de Recuperação", f"{taxa_recuperacao:.2%}")
            
            valor_recuperado = st.session_state.collection_data.loc[
                st.session_state.collection_data['sucesso_cobranca'], 'valor_atraso'
            ].sum()
            st.metric("Valor Recuperado", f"R$ {valor_recuperado:,.2f}")
        
        # Gráficos de distribuição de resultados
        st.subheader("Distribuição dos Resultados de Cobrança")
        
        # Gráfico de pizza para resultados
        results_count = st.session_state.collection_data['resultado_texto'].value_counts()
        
        fig = px.pie(
            values=results_count.values, 
            names=results_count.index,
            title="Resultados de Cobrança",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de barras para estratégias x resultados
        strategy_results = pd.crosstab(
            st.session_state.collection_data['estrategia_cobranca'],
            st.session_state.collection_data['resultado_texto'],
            normalize='index'
        ) * 100  # para percentual
        
        fig = px.bar(
            strategy_results, 
            barmode='group',
            title="Efetividade das Estratégias de Cobrança",
            labels={'value': 'Percentual (%)', 'estrategia_cobranca': 'Estratégia'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mapa de calor para correlações
        st.subheader("Correlações entre Variáveis de Cobrança")
        
        # Selecionar colunas relevantes para correlação
        corr_cols = [
            'dias_atraso', 'valor_atraso', 'contatos_realizados', 
            'resposta_contato', 'proposta_negociacao', 'resultado',
            'idade', 'renda_anual', 'tempo_emprego', 'divida_total'
        ]
        
        corr_matrix = st.session_state.collection_data[corr_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            annot=True,
            fmt=".2f"
        )
        plt.title('Matriz de Correlação', fontsize=16)
        st.pyplot(fig)

    with tab2:
        st.subheader("Análise de Estratégias de Cobrança")
        
        # Filtros para análise
        st.write("**Filtros para análise**")
        col1, col2 = st.columns(2)
        
        with col1:
            filter_estrategia = st.multiselect(
                "Estratégia de Cobrança:",
                options=st.session_state.collection_data['estrategia_cobranca'].unique(),
                default=st.session_state.collection_data['estrategia_cobranca'].unique()
            )
        
        with col2:
            filter_valor = st.slider(
                "Faixa de Valor em Atraso (R$):",
                float(st.session_state.collection_data['valor_atraso'].min()),
                float(st.session_state.collection_data['valor_atraso'].max()),
                (float(st.session_state.collection_data['valor_atraso'].min()), 
                 float(st.session_state.collection_data['valor_atraso'].max()))
            )
        
        # Filtrar dados
        filtered_data = st.session_state.collection_data[
            (st.session_state.collection_data['estrategia_cobranca'].isin(filter_estrategia)) &
            (st.session_state.collection_data['valor_atraso'] >= filter_valor[0]) &
            (st.session_state.collection_data['valor_atraso'] <= filter_valor[1])
        ]
        
        if len(filtered_data) == 0:
            st.warning("Nenhum registro encontrado com os filtros selecionados.")
        else:
            # Análise de taxas de sucesso por estratégia
            st.subheader("Taxa de Sucesso por Estratégia")
            
            strategy_success = filtered_data.groupby('estrategia_cobranca')['sucesso_cobranca'].mean().reset_index()
            strategy_success.columns = ['Estratégia', 'Taxa de Sucesso']
            strategy_success['Taxa de Sucesso'] = strategy_success['Taxa de Sucesso'] * 100
            
            # Ordenar do mais efetivo para o menos
            strategy_success = strategy_success.sort_values('Taxa de Sucesso', ascending=False)
            
            fig = px.bar(
                strategy_success,
                x='Estratégia',
                y='Taxa de Sucesso',
                title="Taxa de Sucesso por Estratégia (%)",
                labels={'Taxa de Sucesso': 'Taxa de Sucesso (%)'},
                color='Taxa de Sucesso',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise por faixa de valor
            st.subheader("Sucesso por Faixa de Valor")
            
            # Criar bins para valores
            filtered_data['faixa_valor'] = pd.cut(
                filtered_data['valor_atraso'],
                bins=[0, 2000, 5000, 10000, 20000],
                labels=['Até R$2.000', 'R$2.001 - R$5.000', 'R$5.001 - R$10.000', 'Acima de R$10.000']
            )
            
            # Análise cruzada de estratégia x faixa de valor
            value_strategy = pd.crosstab(
                filtered_data['faixa_valor'],
                filtered_data['estrategia_cobranca'],
                values=filtered_data['sucesso_cobranca'],
                aggfunc='mean'
            ) * 100
            
            # Gráfico de calor
            fig = px.imshow(
                value_strategy,
                title="Taxa de Sucesso por Estratégia e Faixa de Valor (%)",
                labels=dict(x="Estratégia", y="Faixa de Valor", color="Taxa de Sucesso (%)"),
                color_continuous_scale='Viridis',
                text_auto='.1f'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise por dias de atraso
            st.subheader("Sucesso por Dias de Atraso")
            
            filtered_data['faixa_atraso'] = pd.cut(
                filtered_data['dias_atraso'],
                bins=[0, 30, 60, 90, 180],
                labels=['Até 30 dias', '31-60 dias', '61-90 dias', 'Mais de 90 dias']
            )
            
            # Taxa de sucesso por faixa de atraso
            atraso_success = filtered_data.groupby('faixa_atraso')['sucesso_cobranca'].mean().reset_index()
            atraso_success.columns = ['Faixa de Atraso', 'Taxa de Sucesso']
            atraso_success['Taxa de Sucesso'] = atraso_success['Taxa de Sucesso'] * 100
            
            fig = px.bar(
                atraso_success,
                x='Faixa de Atraso',
                y='Taxa de Sucesso',
                title="Taxa de Sucesso por Faixa de Atraso (%)",
                color='Taxa de Sucesso',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise cruzada de estratégia x faixa de atraso
            atraso_strategy = pd.crosstab(
                filtered_data['faixa_atraso'],
                filtered_data['estrategia_cobranca'],
                values=filtered_data['sucesso_cobranca'],
                aggfunc='mean'
            ) * 100
            
            # Gráfico de calor
            fig = px.imshow(
                atraso_strategy,
                title="Taxa de Sucesso por Estratégia e Faixa de Atraso (%)",
                labels=dict(x="Estratégia", y="Faixa de Atraso", color="Taxa de Sucesso (%)"),
                color_continuous_scale='Viridis',
                text_auto='.1f'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise de proposta de negociação
            st.subheader("Impacto da Proposta de Negociação")
            
            proposta_impact = filtered_data.groupby(['estrategia_cobranca', 'proposta_negociacao'])['sucesso_cobranca'].mean().reset_index()
            proposta_impact['proposta_negociacao'] = proposta_impact['proposta_negociacao'].map({0: 'Sem Proposta', 1: 'Com Proposta'})
            proposta_impact['Taxa de Sucesso'] = proposta_impact['sucesso_cobranca'] * 100
            
            fig = px.bar(
                proposta_impact,
                x='estrategia_cobranca',
                y='Taxa de Sucesso',
                color='proposta_negociacao',
                barmode='group',
                title="Impacto da Proposta de Negociação por Estratégia",
                labels={
                    'estrategia_cobranca': 'Estratégia',
                    'proposta_negociacao': 'Proposta de Negociação'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Conclusões e insights
            st.subheader("Insights sobre Estratégias")
            
            # Determinar a melhor estratégia geral
            best_strategy = strategy_success.iloc[0]['Estratégia']
            best_rate = strategy_success.iloc[0]['Taxa de Sucesso']
            
            # Determinar a melhor estratégia por faixa de valor
            best_by_value = {}
            for faixa in value_strategy.index:
                best = value_strategy.loc[faixa].idxmax()
                best_by_value[faixa] = best
            
            # Determinar a melhor estratégia por faixa de atraso
            best_by_days = {}
            for faixa in atraso_strategy.index:
                best = atraso_strategy.loc[faixa].idxmax()
                best_by_days[faixa] = best
            
            # Impacto da proposta
            proposta_diff = proposta_impact.pivot(
                index='estrategia_cobranca', 
                columns='proposta_negociacao', 
                values='Taxa de Sucesso'
            )
            proposta_diff['diferenca'] = proposta_diff['Com Proposta'] - proposta_diff['Sem Proposta']
            best_proposta_impact = proposta_diff['diferenca'].idxmax()
            
            # Exibir insights
            st.write(f"""
            #### Principais descobertas:
            
            1. **Estratégia mais efetiva em geral**: {best_strategy} (taxa de sucesso de {best_rate:.1f}%)
            
            2. **Estratégia recomendada por faixa de valor**:
               - {best_by_value.get('Até R$2.000', 'N/A')} para valores até R$2.000
               - {best_by_value.get('R$2.001 - R$5.000', 'N/A')} para valores entre R$2.001 e R$5.000
               - {best_by_value.get('R$5.001 - R$10.000', 'N/A')} para valores entre R$5.001 e R$10.000
               - {best_by_value.get('Acima de R$10.000', 'N/A')} para valores acima de R$10.000
            
            3. **Estratégia recomendada por tempo de atraso**:
               - {best_by_days.get('Até 30 dias', 'N/A')} para atrasos até 30 dias
               - {best_by_days.get('31-60 dias', 'N/A')} para atrasos entre 31 e 60 dias
               - {best_by_days.get('61-90 dias', 'N/A')} para atrasos entre 61 e 90 dias
               - {best_by_days.get('Mais de 90 dias', 'N/A')} para atrasos acima de 90 dias
            
            4. **Impacto da proposta de negociação**: A estratégia {best_proposta_impact} apresenta o maior ganho de efetividade quando inclui uma proposta de negociação.
            """)
            
            # Recomendações
            st.info("""
            💡 **Recomendações para otimizar a recuperação**:
            
            1. **Estratégias personalizadas**: Adequar a estratégia de cobrança conforme o perfil da dívida (valor e tempo)
            2. **Priorizar propostas de negociação**: Incluir propostas de negociação aumenta significativamente as taxas de recuperação
            3. **Contato precoce**: A taxa de recuperação cai substancialmente após 90 dias de atraso
            4. **Segmentação de carteira**: Dividir a carteira por faixas de valor e aplicar estratégias específicas
            """)

    with tab3:
        st.subheader("Simulador de Recuperação de Crédito")
        
        st.write("""
        Este simulador permite prever a probabilidade de recuperação de um crédito 
        específico e testar diferentes estratégias para otimizar as chances de sucesso.
        """)
        
        # Dados do cliente/dívida
        st.subheader("Dados do Cliente e da Dívida")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Características do cliente
            cliente_idade = st.slider("Idade do Cliente:", 18, 75, 35)
            cliente_renda = st.slider("Renda Anual (R$):", 0, 150000, 60000, 5000)
            cliente_tempo_emprego = st.slider("Tempo de Emprego (anos):", 0, 40, 5)
            cliente_tempo_residencia = st.slider("Tempo de Residência (anos):", 0, 20, 3)
        
        with col2:
            # Características da dívida
            cliente_valor_atraso = st.slider("Valor em Atraso (R$):", 500, 20000, 5000, 500)
            cliente_dias_atraso = st.slider("Dias em Atraso:", 10, 180, 45)
            cliente_qtd_contas = st.slider("Quantidade de Contas:", 1, 10, 2)
            cliente_atrasos_anteriores = st.slider("Atrasos Anteriores:", 0, 12, 1)
        
        # Estratégia de cobrança
        st.subheader("Estratégia de Cobrança")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cliente_estrategia = st.selectbox(
                "Estratégia de Contato:",
                options=['SMS', 'E-mail', 'Ligação', 'Carta', 'Visita']
            )
            
            cliente_contatos = st.slider(
                "Número de Tentativas de Contato:",
                1, 10, 3
            )
        
        with col2:
            cliente_proposta = st.radio(
                "Incluir Proposta de Negociação?",
                options=["Sim", "Não"],
                horizontal=True
            )
            
            cliente_desconto = st.slider(
                "Percentual de Desconto Oferecido (%):",
                0, 50, 15, 5,
                disabled=(cliente_proposta == "Não")
            )
        
        # Calculadora de probabilidade
        if st.button("Calcular Probabilidade de Recuperação", type="primary"):
            # Criar características para o modelo
            base_proba = 0.3  # probabilidade base
            
            # Ajustes baseados nas características
            prob_ajuste = (
                - 0.001 * cliente_dias_atraso 
                + 0.02 * (cliente_contatos > 3)
                + 0.15 * (cliente_proposta == "Sim")
                - 0.00005 * cliente_valor_atraso
                + 0.0000005 * cliente_renda
                + 0.005 * cliente_tempo_emprego
                - 0.05 * cliente_atrasos_anteriores
                + 0.01 * (cliente_desconto / 10)
            )
            
            # Ajuste pela estratégia
            estrategia_bonus = {
                'SMS': 0.05,
                'E-mail': 0.02,
                'Ligação': 0.15,
                'Carta': 0.08,
                'Visita': 0.25
            }
            
            prob_ajuste += estrategia_bonus[cliente_estrategia]
            
            # Probabilidade final
            prob_sucesso = np.clip(base_proba + prob_ajuste, 0.01, 0.95)
            
            # Exibir resultado
            st.subheader("Resultado da Análise")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gauge para probabilidade
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_sucesso * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilidade de Recuperação (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps' : [
                            {'range': [0, 30], 'color': "red"},
                            {'range': [30, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': prob_sucesso * 100
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # Resultado estimado
                resultado_esperado = "Recuperação Incerta"
                if prob_sucesso < 0.3:
                    resultado_esperado = "Baixa Chance de Recuperação"
                    resultado_color = "red"
                elif prob_sucesso < 0.6:
                    resultado_esperado = "Recuperação Possível"
                    resultado_color = "orange"
                else:
                    resultado_esperado = "Alta Chance de Recuperação"
                    resultado_color = "green"
                
                st.markdown(f"### Resultado: <span style='color:{resultado_color}'>{resultado_esperado}</span>", unsafe_allow_html=True)
            
            with col2:
                # Impacto de cada fator
                st.subheader("Fatores de Impacto")
                
                # Lista de fatores e seus impactos
                fatores = [
                    {"fator": "Estratégia de contato", "impacto": estrategia_bonus[cliente_estrategia] * 100},
                    {"fator": "Dias em atraso", "impacto": -0.1 * cliente_dias_atraso},
                    {"fator": "Proposta de negociação", "impacto": 15 if cliente_proposta == "Sim" else 0},
                    {"fator": "Número de contatos", "impacto": 2 * cliente_contatos},
                    {"fator": "Desconto oferecido", "impacto": cliente_desconto * 0.3 if cliente_proposta == "Sim" else 0},
                    {"fator": "Histórico de atrasos", "impacto": -5 * cliente_atrasos_anteriores}
                ]
                
                # Ordenar por impacto absoluto
                fatores_df = pd.DataFrame(fatores)
                fatores_df['abs_impacto'] = fatores_df['impacto'].abs()
                fatores_df = fatores_df.sort_values('abs_impacto', ascending=False).head(5)
                
                # Criar gráfico de barras horizontais
                fatores_df['cor'] = fatores_df['impacto'].apply(
                    lambda x: 'green' if x > 0 else 'red'
                )
                
                fig = px.bar(
                    fatores_df,
                    y='fator',
                    x='impacto',
                    orientation='h',
                    title="Impacto dos Fatores na Probabilidade",
                    labels={'impacto': 'Impacto (%)', 'fator': 'Fator'},
                    color='cor',
                    color_discrete_map={'green': 'green', 'red': 'red'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recomendações para aumentar a chance de recuperação
            st.subheader("Recomendações para Aumentar a Chance de Recuperação")
            
            recomendacoes = []
            
            if cliente_proposta == "Não":
                recomendacoes.append("• **Fazer uma proposta de negociação** pode aumentar significativamente as chances de recuperação.")
            
            if cliente_contatos < 5:
                recomendacoes.append("• **Aumentar o número de tentativas de contato** para pelo menos 5 pode melhorar o engajamento.")
            
            if cliente_estrategia != 'Visita' and cliente_valor_atraso > 10000:
                recomendacoes.append("• Para valores altos (>R$10.000), considere usar **visita presencial** como estratégia.")
            
            if cliente_estrategia != 'Ligação' and cliente_dias_atraso > 60:
                recomendacoes.append("• Para atrasos longos (>60 dias), **ligações diretas** tendem a ser mais efetivas.")
            
            if cliente_desconto < 20 and cliente_proposta == "Sim" and cliente_dias_atraso > 90:
                recomendacoes.append("• Para dívidas antigas (>90 dias), considere **aumentar o desconto oferecido** para pelo menos 25%.")
            
            if cliente_dias_atraso > 120:
                recomendacoes.append("• Esta dívida está em atraso há muito tempo. Considere **terceirizar a cobrança** ou oferecer um **desconto significativo** para recuperação parcial.")
            
            if not recomendacoes:
                recomendacoes.append("• Sua estratégia atual está bem otimizada para este caso específico.")
            
            for rec in recomendacoes:
                st.markdown(rec)
            
            # Comparação com estratégias alternativas
            st.subheader("Comparação com Estratégias Alternativas")
            
            # Criar DataFrame de comparação
            estrategias = ['SMS', 'Email', 'Ligação', 'Carta', 'Visita']
            propostas = ['Com proposta', 'Sem proposta']
            
            comparison_data = []
            
            # Estratégia atual (para destacar)
            estrategia_atual = cliente_estrategia
            proposta_atual = "Com proposta" if cliente_proposta == "Sim" else "Sem proposta"
            
            for e in estrategias:
                for p in propostas:
                    # Calcular probabilidade para esta combinação
                    temp_prob = base_proba
                    
                    # Ajuste para estratégia
                    if e == 'SMS':
                        temp_prob *= 0.8
                    elif e == 'Email':
                        temp_prob *= 0.9
                    elif e == 'Ligação':
                        temp_prob *= 1.2
                    elif e == 'Carta':
                        temp_prob *= 0.7
                    elif e == 'Visita':
                        temp_prob *= 1.5
                    
                    # Ajuste para proposta
                    if p == 'Com proposta':
                        temp_prob *= 1.4
                    
                    # Verificar se é a estratégia atual
                    is_current = (e == estrategia_atual) and (p == proposta_atual)
                    
                    comparison_data.append({
                        'Estratégia': e,
                        'Proposta': p,
                        'Probabilidade': min(temp_prob * 100, 99.9),
                        'Atual': is_current
                    })
            
            # Criar DataFrame e ordenar por probabilidade
            comparison_df = pd.DataFrame(comparison_data).sort_values('Probabilidade', ascending=False)
            
            # Função para destacar a estratégia atual
            def highlight_current(s):
                is_current = s['Atual']
                return ['background-color: #a8d08d' if is_current else '' for _ in s]
            
            # Criar DataFrame para exibição (sem a coluna 'Atual')
            display_df = comparison_df[['Estratégia', 'Proposta', 'Probabilidade']]
            
            # Aplicar formatação e exibir
            st.dataframe(
                display_df.style.apply(
                    lambda row: ['background-color: #a8d08d' if comparison_df.iloc[row.name]['Atual'] else '' for _ in row],
                    axis=1
                ),
                use_container_width=True
            )
            
            # Mostrar estratégia ótima
            best_strategy = comparison_df.iloc[0]
            
            if not best_strategy['Atual']:
                st.success(f"""
                **Estratégia Ótima Recomendada**: {best_strategy['Estratégia']} com 
                {best_strategy['Proposta']} para proposta de negociação.
                
                Esta combinação poderia aumentar a probabilidade de recuperação para 
                {best_strategy['Probabilidade']:.1f}%, um ganho de 
                {best_strategy['Probabilidade'] - prob_sucesso * 100:.1f} pontos percentuais.
                """)
            else:
                st.success("Você já está usando a estratégia ótima para este caso!")

# Sidebar com opções do app
st.sidebar.title("CreditWise")
st.sidebar.image("https://www.svgrepo.com/show/530453/financial-profit.svg", width=100)

app_mode = st.sidebar.selectbox(
    "Selecione o Módulo",
    ["Visão Geral", "Modelo de Crédito", "Análise de Cobrança", "Previsão de Fluxo de Caixa", "Simulação de Portfólio", "Simulador de Crédito"]
)

# Visão Geral
if app_mode == "Visão Geral":
    st.title("CreditWise - Análise de Risco e Cobrança 💳")
    # ... código existente ...

# Modelo de Crédito
elif app_mode == "Modelo de Crédito":
    # ... código existente ...
    pass

# Análise de Cobrança
elif app_mode == "Análise de Cobrança":
    # ... código existente ...
    pass

# Previsão de Fluxo de Caixa
elif app_mode == "Previsão de Fluxo de Caixa":
    st.title("Previsão de Fluxo de Caixa 📈")
    st.write("""
    Este módulo permite prever as receitas futuras baseadas na recuperação de crédito prevista.
    Visualize o impacto das estratégias de recuperação no seu fluxo de caixa ao longo do tempo.
    """)
    
    # Parâmetros de entrada
    st.header("Parâmetros de Previsão")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Parâmetros relacionados à carteira
        st.subheader("Carteira de Crédito")
        
        total_portfolio = st.number_input(
            "Valor Total da Carteira (R$)",
            min_value=10000.0,
            max_value=10000000.0,
            value=1000000.0,
            step=50000.0,
            format="%.2f"
        )
        
        num_accounts = st.number_input(
            "Número de Contas",
            min_value=10,
            max_value=10000,
            value=500,
            step=10
        )
        
        avg_account_value = total_portfolio / num_accounts
        st.metric("Valor Médio por Conta", f"R$ {avg_account_value:.2f}")
        
        portfolio_age = st.slider(
            "Idade Média da Carteira (dias)",
            min_value=30,
            max_value=365,
            value=120,
            step=15
        )
        
    with col2:
        # Parâmetros de estratégia de recuperação
        st.subheader("Estratégia de Recuperação")
        
        recovery_rate = st.slider(
            "Taxa de Recuperação Esperada (%)",
            min_value=10.0,
            max_value=90.0,
            value=40.0,
            step=5.0
        )
        
        recovery_time = st.slider(
            "Tempo para Recuperação (meses)",
            min_value=1,
            max_value=24,
            value=6,
            step=1
        )
        
        recovery_curve = st.selectbox(
            "Curva de Recuperação",
            ["Linear", "Exponencial Decrescente", "Sigmoide", "Personalizada"]
        )
        
        operational_costs = st.slider(
            "Custos Operacionais (%)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=2.5
        )
    
    # Aplicar simulação
    if st.button("Gerar Previsão de Fluxo de Caixa"):
        # Calcular previsão de fluxo
        st.header("Previsão de Fluxo de Caixa")
        
        # Criar dataframe com previsão
        months = list(range(1, recovery_time + 1))
        recovery_data = []
        
        # Calcular valores para cada mês baseado na curva selecionada
        total_recovery_value = total_portfolio * (recovery_rate / 100)
        
        if recovery_curve == "Linear":
            monthly_values = [total_recovery_value / recovery_time] * recovery_time
        
        elif recovery_curve == "Exponencial Decrescente":
            # Peso maior nos primeiros meses
            factor = 1.5
            weights = [np.exp(-factor * i / recovery_time) for i in range(recovery_time)]
            total_weight = sum(weights)
            monthly_values = [(w / total_weight) * total_recovery_value for w in weights]
            
        elif recovery_curve == "Sigmoide":
            # Recuperação concentrada no meio do período
            center = recovery_time / 2
            steepness = 5 / recovery_time
            weights = [1 / (1 + np.exp(-steepness * (i - center))) - 
                       1 / (1 + np.exp(-steepness * ((i-1) - center))) for i in range(1, recovery_time + 1)]
            # Ajustar para garantir que soma seja 1
            total_weight = sum(weights)
            monthly_values = [(w / total_weight) * total_recovery_value for w in weights]
        
        else:  # Personalizada - exemplo simples
            # Mais nos meses 2-4
            custom_weights = [0.1, 0.2, 0.25, 0.2, 0.15, 0.1]
            if len(custom_weights) < recovery_time:
                custom_weights.extend([0.05] * (recovery_time - len(custom_weights)))
            custom_weights = custom_weights[:recovery_time]  # Truncar se for maior
            # Normalizar pesos
            total_weight = sum(custom_weights)
            normalized_weights = [w / total_weight for w in custom_weights]
            monthly_values = [w * total_recovery_value for w in normalized_weights]
        
        # Calcular acumulado e outros dados para análise
        cumulative_recovery = 0
        
        for i, month in enumerate(months):
            monthly_recovery = monthly_values[i]
            cumulative_recovery += monthly_recovery
            monthly_cost = monthly_recovery * (operational_costs / 100)
            monthly_net = monthly_recovery - monthly_cost
            
            recovery_data.append({
                'Mês': month,
                'Recuperação Mensal': monthly_recovery,
                'Recuperação Acumulada': cumulative_recovery,
                'Custos Operacionais': monthly_cost,
                'Líquido Mensal': monthly_net,
                'Líquido Acumulado': cumulative_recovery - (monthly_recovery * (operational_costs / 100) * month)
            })
        
        df_recovery = pd.DataFrame(recovery_data)
        
        # Gráficos e visualizações
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de recuperação mensal vs acumulada
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_recovery['Mês'],
                y=df_recovery['Recuperação Mensal'],
                name='Recuperação Mensal',
                marker_color='#3366CC'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_recovery['Mês'],
                y=df_recovery['Recuperação Acumulada'],
                name='Recuperação Acumulada',
                marker_color='#FF9900',
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title='Recuperação Mensal vs. Acumulada',
                xaxis_title='Mês',
                yaxis_title='Valor (R$)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gráfico de fluxo líquido
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_recovery['Mês'],
                y=df_recovery['Líquido Mensal'],
                name='Líquido Mensal',
                marker_color='#109618'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_recovery['Mês'],
                y=df_recovery['Líquido Acumulado'],
                name='Líquido Acumulado',
                marker_color='#DC3912',
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title='Fluxo de Caixa Líquido',
                xaxis_title='Mês',
                yaxis_title='Valor (R$)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Indicadores de performance
        st.subheader("Indicadores de Performance")
        
        # KPIs em colunas
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            # Total recuperado
            final_recovery = df_recovery['Recuperação Acumulada'].iloc[-1]
            recovery_percent = (final_recovery / total_portfolio) * 100
            
            st.metric(
                "Total Recuperado", 
                f"R$ {final_recovery:,.2f}",
                f"{recovery_percent:.1f}% da carteira"
            )
        
        with kpi2:
            # Líquido após custos
            final_net = df_recovery['Líquido Acumulado'].iloc[-1]
            net_percent = (final_net / total_portfolio) * 100
            
            st.metric(
                "Líquido Final", 
                f"R$ {final_net:,.2f}",
                f"{net_percent:.1f}% da carteira"
            )
            
        with kpi3:
            # Tempo para recuperar 50%
            half_recovery = total_recovery_value * 0.5
            months_to_half = None
            
            for i, row in df_recovery.iterrows():
                if row['Recuperação Acumulada'] >= half_recovery:
                    months_to_half = row['Mês']
                    break
            
            if months_to_half:
                st.metric(
                    "Prazo para 50% da Recuperação", 
                    f"{months_to_half} meses"
                )
            else:
                st.metric("Prazo para 50% da Recuperação", "N/A")
        
        with kpi4:
            # ROI
            total_cost = df_recovery['Custos Operacionais'].sum()
            roi = ((final_recovery - total_cost) / total_cost) * 100
            
            st.metric(
                "ROI da Operação", 
                f"{roi:.1f}%"
            )
        
        # Tabela com os dados mensais
        st.subheader("Detalhamento Mensal")
        
        display_df = df_recovery.copy()
        
        # Formatar para visualização
        display_cols = {
            'Mês': 'Mês',
            'Recuperação Mensal': 'Recuperação Mensal (R$)',
            'Recuperação Acumulada': 'Recuperação Acumulada (R$)',
            'Custos Operacionais': 'Custos Operacionais (R$)',
            'Líquido Mensal': 'Líquido Mensal (R$)',
            'Líquido Acumulado': 'Líquido Acumulado (R$)'
        }
        
        display_df = display_df.rename(columns=display_cols)
        
        # Formatar valores monetários
        for col in display_df.columns:
            if 'R$' in col:
                display_df[col] = display_df[col].apply(lambda x: f"R$ {x:,.2f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Análise de sensibilidade
        st.subheader("Análise de Sensibilidade")
        st.write("Veja como diferentes taxas de recuperação afetam o resultado final.")
        
        # Criar dados para análise de sensibilidade
        sensitivity_rates = [recovery_rate - 20, recovery_rate - 10, recovery_rate, recovery_rate + 10, recovery_rate + 20]
        sensitivity_rates = [max(10, min(rate, 90)) for rate in sensitivity_rates]  # Limitar entre 10% e 90%
        
        sensitivity_data = []
        
        for rate in sensitivity_rates:
            recovery_value = total_portfolio * (rate / 100)
            net_value = recovery_value * (1 - operational_costs / 100)
            
            sensitivity_data.append({
                'Taxa de Recuperação (%)': rate,
                'Valor Recuperado (R$)': recovery_value,
                'Valor Líquido (R$)': net_value,
                'Percentual da Carteira (%)': (recovery_value / total_portfolio) * 100
            })
        
        df_sensitivity = pd.DataFrame(sensitivity_data)
        
        # Gráfico de sensibilidade
        fig = px.bar(
            df_sensitivity, 
            x='Taxa de Recuperação (%)', 
            y='Valor Recuperado (R$)',
            color='Percentual da Carteira (%)',
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={'Valor Recuperado (R$)': 'Valor Recuperado (R$)'},
            text_auto='.2s'
        )
        
        fig.update_layout(
            title='Análise de Sensibilidade por Taxa de Recuperação',
            xaxis_title='Taxa de Recuperação (%)',
            yaxis_title='Valor Recuperado (R$)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recomendações
        st.subheader("Recomendações")
        
        st.markdown(f"""
        ### Com base na análise de fluxo de caixa, recomendamos:
        
        - **Alocação de Recursos**: Concentrar esforços de cobrança nos {months_to_half} primeiros meses, quando a taxa de recuperação é mais alta.
        
        - **Investimento em Estratégias**: Um investimento adicional de até {(operational_costs * 0.25):.1f}% em custos operacionais pode ser justificado se aumentar a taxa de recuperação em pelo menos {(operational_costs * 0.5):.1f}%.
        
        - **Prazo de Retorno**: O ponto de equilíbrio será atingido no mês {max(1, int(operational_costs / 10))}, com recuperação superando custos acumulados.
        
        - **Risco de Carteira**: Diversificar a carteira para reduzir concentração e volatilidade nos fluxos mensais.
        """)
        
        st.info("Esta análise é baseada em projeções estatísticas. Os resultados reais podem variar dependendo de fatores externos e específicos de cada cliente.")

# Simulador de Crédito
elif app_mode == "Simulador de Crédito":
    # ... código existente ...
    pass

# Simulação de Portfólio
elif app_mode == "Simulação de Portfólio":
    st.title("Simulação de Portfólio de Crédito 📊")
    st.write("""
    Este módulo permite simular diferentes composições de carteira de crédito, analisar o impacto de estratégias 
    de concessão e estimar como diversos cenários econômicos afetam o desempenho do portfólio.
    """)
    
    # Tabs para as diferentes funcionalidades
    portfolio_tab = st.tabs(["Composição de Carteira", "Análise de Estratégias", "Cenários Econômicos"])
    
    # Tab 1: Composição de Carteira
    with portfolio_tab[0]:
        st.subheader("Simulação de Composição de Carteira")
        st.write("Configure diferentes perfis de risco para sua carteira e analise o impacto no retorno e inadimplência.")
        
        # Parâmetros do portfólio
        st.write("### Configuração da Carteira")
        
        # Definir o tamanho total da carteira
        total_portfolio_size = st.number_input(
            "Valor Total da Carteira (R$)",
            min_value=100000.0,
            max_value=100000000.0,
            value=10000000.0,
            step=1000000.0,
            format="%.2f"
        )
        
        # Configurar os segmentos da carteira
        st.write("### Distribuição por Perfil de Risco")
        st.write("Defina a proporção da carteira alocada para cada perfil de risco:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição entre perfis de risco
            risk_low = st.slider("Baixo Risco (%)", 0, 100, 30, 5)
            risk_medium = st.slider("Médio Risco (%)", 0, 100, 40, 5)
            risk_high = st.slider("Alto Risco (%)", 0, 100, 30, 5)
            
            # Verificar se a soma é 100%
            total_risk = risk_low + risk_medium + risk_high
            if total_risk != 100:
                st.warning(f"A soma dos percentuais deve ser 100%. Atualmente: {total_risk}%")
            
            # Calcular valores por segmento
            low_value = total_portfolio_size * (risk_low / 100)
            medium_value = total_portfolio_size * (risk_medium / 100)
            high_value = total_portfolio_size * (risk_high / 100)
            
            # Mostrar distribuição em valores
            st.write("### Valor por Segmento")
            st.metric("Baixo Risco", f"R$ {low_value:,.2f}")
            st.metric("Médio Risco", f"R$ {medium_value:,.2f}")
            st.metric("Alto Risco", f"R$ {high_value:,.2f}")
        
        with col2:
            # Parâmetros por perfil de risco
            st.write("### Características por Perfil")
            
            # Taxas de juros por perfil
            st.write("**Taxas de Juros Anuais**")
            interest_low = st.slider("Taxa para Baixo Risco (%)", 5.0, 30.0, 12.0, 0.5)
            interest_medium = st.slider("Taxa para Médio Risco (%)", 10.0, 50.0, 24.0, 0.5)
            interest_high = st.slider("Taxa para Alto Risco (%)", 15.0, 80.0, 36.0, 0.5)
            
            # Taxas de inadimplência por perfil
            st.write("**Taxas de Inadimplência Esperadas**")
            default_low = st.slider("Inadimplência em Baixo Risco (%)", 0.5, 10.0, 2.0, 0.5)
            default_medium = st.slider("Inadimplência em Médio Risco (%)", 3.0, 20.0, 8.0, 0.5)
            default_high = st.slider("Inadimplência em Alto Risco (%)", 10.0, 40.0, 25.0, 0.5)
        
        # Botão para simular
        if st.button("Simular Composição de Carteira"):
            # Cálculos para a simulação
            
            # Receita esperada por segmento (juros)
            revenue_low = low_value * (interest_low / 100)
            revenue_medium = medium_value * (interest_medium / 100)
            revenue_high = high_value * (interest_high / 100)
            total_revenue = revenue_low + revenue_medium + revenue_high
            
            # Perdas esperadas por inadimplência
            loss_low = low_value * (default_low / 100)
            loss_medium = medium_value * (default_medium / 100)
            loss_high = high_value * (default_high / 100)
            total_loss = loss_low + loss_medium + loss_high
            
            # Resultado líquido
            net_low = revenue_low - loss_low
            net_medium = revenue_medium - loss_medium
            net_high = revenue_high - loss_high
            total_net = total_revenue - total_loss
            
            # ROI por segmento
            roi_low = (net_low / low_value) * 100
            roi_medium = (net_medium / medium_value) * 100
            roi_high = (net_high / high_value) * 100
            roi_total = (total_net / total_portfolio_size) * 100
            
            # Criar DataFrame com os resultados
            portfolio_data = {
                'Segmento': ['Baixo Risco', 'Médio Risco', 'Alto Risco', 'Total'],
                'Valor da Carteira (R$)': [low_value, medium_value, high_value, total_portfolio_size],
                'Receita Esperada (R$)': [revenue_low, revenue_medium, revenue_high, total_revenue],
                'Perda Esperada (R$)': [loss_low, loss_medium, loss_high, total_loss],
                'Resultado Líquido (R$)': [net_low, net_medium, net_high, total_net],
                'ROI (%)': [roi_low, roi_medium, roi_high, roi_total],
                'Inadimplência (%)': [default_low, default_medium, default_high, (total_loss/total_portfolio_size)*100]
            }
            
            df_portfolio = pd.DataFrame(portfolio_data)
            
            # Exibir resultados
            st.write("## Resultados da Simulação")
            
            # KPIs principais
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            with kpi1:
                st.metric("Receita Total", f"R$ {total_revenue:,.2f}")
            
            with kpi2:
                st.metric("Perda por Inadimplência", f"R$ {total_loss:,.2f}")
            
            with kpi3:
                st.metric("Resultado Líquido", f"R$ {total_net:,.2f}")
            
            with kpi4:
                st.metric("ROI da Carteira", f"{roi_total:.2f}%")
            
            # Gráficos
            st.write("### Análise Visual")
            
            tab1, tab2, tab3 = st.tabs(["Composição", "Performance", "Risco-Retorno"])
            
            with tab1:
                # Gráfico de composição da carteira
                fig = px.pie(
                    df_portfolio[:-1],  # Excluir linha de total
                    values='Valor da Carteira (R$)',
                    names='Segmento',
                    title='Composição da Carteira por Segmento',
                    color='Segmento',
                    color_discrete_map={
                        'Baixo Risco': '#2ECC71',
                        'Médio Risco': '#F39C12',
                        'Alto Risco': '#E74C3C'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Gráfico de barras comparando receita, perda e resultado líquido
                fig = go.Figure()
                
                segments = df_portfolio['Segmento'][:-1]  # Excluir linha de total
                
                fig.add_trace(go.Bar(
                    x=segments,
                    y=df_portfolio['Receita Esperada (R$)'][:-1],
                    name='Receita',
                    marker_color='#3498DB'
                ))
                
                fig.add_trace(go.Bar(
                    x=segments,
                    y=df_portfolio['Perda Esperada (R$)'][:-1],
                    name='Perda',
                    marker_color='#E74C3C'
                ))
                
                fig.add_trace(go.Bar(
                    x=segments,
                    y=df_portfolio['Resultado Líquido (R$)'][:-1],
                    name='Resultado Líquido',
                    marker_color='#2ECC71'
                ))
                
                fig.update_layout(
                    title='Performance por Segmento',
                    xaxis_title='Segmento',
                    yaxis_title='Valor (R$)',
                    barmode='group',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Gráfico de dispersão mostrando relação risco-retorno
                fig = px.scatter(
                    df_portfolio[:-1],  # Excluir linha de total
                    x='Inadimplência (%)',
                    y='ROI (%)',
                    size='Valor da Carteira (R$)',
                    color='Segmento',
                    color_discrete_map={
                        'Baixo Risco': '#2ECC71',
                        'Médio Risco': '#F39C12',
                        'Alto Risco': '#E74C3C'
                    },
                    title='Relação Risco-Retorno por Segmento',
                    labels={
                        'Inadimplência (%)': 'Risco (Taxa de Inadimplência %)',
                        'ROI (%)': 'Retorno (ROI %)'
                    },
                    size_max=60
                )
                
                # Adicionar linha média
                fig.add_shape(
                    type="line",
                    x0=df_portfolio['Inadimplência (%)'].min() - 1,
                    y0=roi_total,
                    x1=df_portfolio['Inadimplência (%)'].max() + 1,
                    y1=roi_total,
                    line=dict(
                        color="gray",
                        width=2,
                        dash="dash",
                    )
                )
                
                fig.update_layout(
                    xaxis_title='Risco (Taxa de Inadimplência %)',
                    yaxis_title='Retorno (ROI %)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Tabela com resultados detalhados
            st.write("### Detalhamento dos Resultados")
            
            # Formatar o DataFrame para exibição
            display_df = df_portfolio.copy()
            format_cols = ['Valor da Carteira (R$)', 'Receita Esperada (R$)', 'Perda Esperada (R$)', 'Resultado Líquido (R$)']
            
            for col in format_cols:
                display_df[col] = display_df[col].apply(lambda x: f"R$ {x:,.2f}")
            
            display_df['ROI (%)'] = display_df['ROI (%)'].apply(lambda x: f"{x:.2f}%")
            display_df['Inadimplência (%)'] = display_df['Inadimplência (%)'].apply(lambda x: f"{x:.2f}%")
            
            # Destacar a linha de total
            def highlight_total(s):
                return ['background-color: #f0f0f0; font-weight: bold' if s.Segmento == 'Total' else '' for _ in s]
            
            # Exibir tabela formatada
            st.dataframe(
                display_df.style.apply(highlight_total, axis=1),
                use_container_width=True
            )
            
            # Recomendações
            st.write("### Recomendações")
            
            recommendations = []
            
            # Avaliar composição da carteira
            if roi_low > roi_medium and roi_low > roi_high:
                recommendations.append("- Considere **aumentar a proporção de Baixo Risco** na carteira, pois este segmento apresenta o melhor ROI.")
            elif roi_medium > roi_low and roi_medium > roi_high:
                recommendations.append("- Considere **aumentar a proporção de Médio Risco** na carteira, pois este segmento apresenta o melhor ROI.")
            elif roi_high > roi_low and roi_high > roi_medium:
                recommendations.append("- Considere **aumentar a proporção de Alto Risco** na carteira, pois este segmento apresenta o melhor ROI, apesar da maior inadimplência.")
            
            # Avaliar taxas de juros
            if roi_low < 5:
                recommendations.append("- As taxas para o segmento de **Baixo Risco** podem estar muito baixas considerando a inadimplência.")
            if roi_medium < 10:
                recommendations.append("- As taxas para o segmento de **Médio Risco** podem estar muito baixas considerando a inadimplência.")
            if roi_high < 15:
                recommendations.append("- As taxas para o segmento de **Alto Risco** podem estar muito baixas considerando a inadimplência.")
            
            # Avaliar equilíbrio risco-retorno
            if roi_total < 5:
                recommendations.append("- O **retorno geral da carteira está baixo**. Considere ajustar taxas de juros ou reduzir exposição a segmentos com maior inadimplência.")
            
            if (total_loss/total_portfolio_size)*100 > 15:
                recommendations.append("- A **perda por inadimplência está alta** (>15% da carteira). Considere melhorar critérios de concessão ou estratégias de cobrança.")
            
            # Exibir recomendações
            if recommendations:
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.success("A composição atual da carteira apresenta um bom equilíbrio entre risco e retorno.")
                
            # Adicionar opção de download dos resultados
            csv = df_portfolio.to_csv(index=False)
            st.download_button(
                label="Download dos Resultados (CSV)",
                data=csv,
                file_name="simulacao_portfolio.csv",
                mime="text/csv",
            )

    # Tab 2: Análise de Estratégias
    with portfolio_tab[1]:
        st.subheader("Impacto de Estratégias de Concessão")
        st.write("Avalie como diferentes critérios de aprovação de crédito impactam o desempenho da carteira.")
        
        # Parâmetros da simulação
        st.write("### Configurações da Simulação")
        
        # Volume de solicitações
        applications_volume = st.number_input(
            "Volume de Solicitações de Crédito (mensal)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100
        )
        
        # Ticket médio
        average_ticket = st.number_input(
            "Ticket Médio (R$)",
            min_value=1000.0,
            max_value=100000.0,
            value=5000.0,
            step=1000.0,
            format="%.2f"
        )
        
        # Distribuição do score de crédito dos solicitantes
        st.write("### Distribuição dos Solicitantes por Score")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Configurar distribuição de solicitantes por faixa de score
            score_very_low = st.slider("Score Muito Baixo (0-300)", 0, 100, 15, 5)
            score_low = st.slider("Score Baixo (301-500)", 0, 100, 25, 5)
            score_medium = st.slider("Score Médio (501-700)", 0, 100, 30, 5)
            score_high = st.slider("Score Alto (701-850)", 0, 100, 20, 5)
            score_very_high = st.slider("Score Muito Alto (851-1000)", 0, 100, 10, 5)
            
            # Verificar se a soma é 100%
            total_score = score_very_low + score_low + score_medium + score_high + score_very_high
            if total_score != 100:
                st.warning(f"A soma dos percentuais deve ser 100%. Atualmente: {total_score}%")
        
        with col2:
            # Parâmetros de inadimplência por faixa de score
            st.write("### Taxa de Inadimplência por Faixa de Score")
            
            default_very_low = st.slider("Inadimplência em Score Muito Baixo (%)", 20.0, 90.0, 60.0, 5.0)
            default_low = st.slider("Inadimplência em Score Baixo (%)", 10.0, 50.0, 30.0, 2.5)
            default_medium = st.slider("Inadimplência em Score Médio (%)", 5.0, 25.0, 15.0, 1.0)
            default_high = st.slider("Inadimplência em Score Alto (%)", 1.0, 15.0, 5.0, 0.5)
            default_very_high = st.slider("Inadimplência em Score Muito Alto (%)", 0.1, 5.0, 1.0, 0.1)
        
        # Configuração de estratégias de aprovação
        st.write("### Estratégias de Aprovação")
        st.write("Configure diferentes estratégias de corte de score para aprovação de crédito:")
        
        # Criar tabs para diferentes estratégias
        strategy_tabs = st.tabs(["Conservadora", "Moderada", "Agressiva", "Personalizada"])
        
        # Definir cut-offs por estratégia
        with strategy_tabs[0]:  # Conservadora
            st.write("#### Estratégia Conservadora")
            st.write("Apenas aprova solicitantes com score alto ou muito alto (> 700)")
            strategy_conservative = {
                'nome': 'Conservadora',
                'muito_baixo': False,
                'baixo': False,
                'medio': False,
                'alto': True,
                'muito_alto': True,
                'taxa_juros': 15.0
            }
            
            st.write("**Taxa de Juros Média:** 15%")
            st.write("**Taxa de Aprovação Esperada:** ~30%")
            
        with strategy_tabs[1]:  # Moderada
            st.write("#### Estratégia Moderada")
            st.write("Aprova solicitantes com score médio ou superior (> 500)")
            strategy_moderate = {
                'nome': 'Moderada',
                'muito_baixo': False,
                'baixo': False,
                'medio': True,
                'alto': True,
                'muito_alto': True,
                'taxa_juros': 22.0
            }
            
            st.write("**Taxa de Juros Média:** 22%")
            st.write("**Taxa de Aprovação Esperada:** ~60%")
            
        with strategy_tabs[2]:  # Agressiva
            st.write("#### Estratégia Agressiva")
            st.write("Aprova solicitantes com score baixo ou superior (> 300)")
            strategy_aggressive = {
                'nome': 'Agressiva',
                'muito_baixo': False,
                'baixo': True,
                'medio': True,
                'alto': True,
                'muito_alto': True,
                'taxa_juros': 30.0
            }
            
            st.write("**Taxa de Juros Média:** 30%")
            st.write("**Taxa de Aprovação Esperada:** ~85%")
            
        with strategy_tabs[3]:  # Personalizada
            st.write("#### Estratégia Personalizada")
            st.write("Configure seus próprios critérios de aprovação e taxa de juros:")
            
            # Configurações personalizadas
            custom_very_low = st.checkbox("Aprovar Score Muito Baixo (0-300)", False)
            custom_low = st.checkbox("Aprovar Score Baixo (301-500)", False)
            custom_medium = st.checkbox("Aprovar Score Médio (501-700)", True)
            custom_high = st.checkbox("Aprovar Score Alto (701-850)", True)
            custom_very_high = st.checkbox("Aprovar Score Muito Alto (851-1000)", True)
            
            custom_interest = st.slider("Taxa de Juros Média (%)", 10.0, 50.0, 25.0, 0.5)
            
            strategy_custom = {
                'nome': 'Personalizada',
                'muito_baixo': custom_very_low,
                'baixo': custom_low,
                'medio': custom_medium,
                'alto': custom_high,
                'muito_alto': custom_very_high,
                'taxa_juros': custom_interest
            }
            
            # Calcular taxa de aprovação esperada
            approval_rate = 0
            if custom_very_low:
                approval_rate += score_very_low
            if custom_low:
                approval_rate += score_low
            if custom_medium:
                approval_rate += score_medium
            if custom_high:
                approval_rate += score_high
            if custom_very_high:
                approval_rate += score_very_high
                
            st.write(f"**Taxa de Aprovação Esperada:** ~{approval_rate}%")
        
        # Botão para comparar estratégias
        if st.button("Comparar Estratégias de Concessão"):
            # Lista com todas as estratégias para comparação
            strategies = [
                strategy_conservative,
                strategy_moderate,
                strategy_aggressive,
                strategy_custom
            ]
            
            # Dataframe para armazenar resultados
            results_data = []
            
            # Loop pelas estratégias
            for strategy in strategies:
                # Calcular volumes e valores
                approved_very_low = applications_volume * (score_very_low / 100) if strategy['muito_baixo'] else 0
                approved_low = applications_volume * (score_low / 100) if strategy['baixo'] else 0
                approved_medium = applications_volume * (score_medium / 100) if strategy['medio'] else 0
                approved_high = applications_volume * (score_high / 100) if strategy['alto'] else 0
                approved_very_high = applications_volume * (score_very_high / 100) if strategy['muito_alto'] else 0
                
                total_approved = approved_very_low + approved_low + approved_medium + approved_high + approved_very_high
                approval_rate = (total_approved / applications_volume) * 100
                
                # Calcular valor da carteira e distribuição
                portfolio_value = total_approved * average_ticket
                
                # Calcular inadimplência esperada por faixa
                default_value_very_low = approved_very_low * average_ticket * (default_very_low / 100)
                default_value_low = approved_low * average_ticket * (default_low / 100)
                default_value_medium = approved_medium * average_ticket * (default_medium / 100)
                default_value_high = approved_high * average_ticket * (default_high / 100)
                default_value_very_high = approved_very_high * average_ticket * (default_very_high / 100)
                
                total_default_value = default_value_very_low + default_value_low + default_value_medium + default_value_high + default_value_very_high
                default_rate = (total_default_value / portfolio_value) * 100 if portfolio_value > 0 else 0
                
                # Calcular receita de juros
                interest_revenue = portfolio_value * (strategy['taxa_juros'] / 100)
                
                # Calcular resultado líquido
                net_result = interest_revenue - total_default_value
                roi = (net_result / portfolio_value) * 100 if portfolio_value > 0 else 0
                
                # Adicionar dados aos resultados
                results_data.append({
                    'Estratégia': strategy['nome'],
                    'Taxa de Aprovação (%)': approval_rate,
                    'Solicitações Aprovadas': total_approved,
                    'Valor da Carteira (R$)': portfolio_value,
                    'Taxa de Inadimplência (%)': default_rate,
                    'Valor de Inadimplência (R$)': total_default_value,
                    'Receita de Juros (R$)': interest_revenue,
                    'Resultado Líquido (R$)': net_result,
                    'ROI (%)': roi
                })
            
            # Criar DataFrame com resultados
            df_results = pd.DataFrame(results_data)
            
            # Exibir resultados
            st.write("## Resultados da Comparação")
            
            # Gráfico de barras comparando as estratégias
            fig = px.bar(
                df_results,
                x='Estratégia',
                y='Resultado Líquido (R$)',
                color='ROI (%)',
                color_continuous_scale='RdYlGn',
                text_auto='.2s',
                title='Comparação do Resultado Líquido por Estratégia'
            )
            
            fig.update_layout(
                xaxis_title='Estratégia de Concessão',
                yaxis_title='Resultado Líquido (R$)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela comparativa
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Gráfico de radar para comparação de múltiplas métricas
                metrics = ['Taxa de Aprovação (%)', 'Taxa de Inadimplência (%)', 'ROI (%)']
                
                fig = go.Figure()
                
                for i, strategy in enumerate(df_results['Estratégia']):
                    values = df_results.loc[i, metrics].tolist()
                    # Adicionar o primeiro valor novamente para fechar o polígono
                    values.append(values[0])
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metrics + [metrics[0]],  # Repetir a primeira categoria
                        fill='toself',
                        name=strategy
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(df_results[metrics].max()) * 1.1]
                        )
                    ),
                    title='Comparação Multidimensional de Estratégias',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Mostrar métricas-chave para a melhor estratégia em termos de ROI
                best_strategy = df_results.loc[df_results['ROI (%)'].idxmax()]
                
                st.write("### Estratégia Ótima")
                st.success(f"**{best_strategy['Estratégia']}**")
                
                st.metric("ROI", f"{best_strategy['ROI (%)']:.2f}%")
                st.metric("Resultado Líquido", f"R$ {best_strategy['Resultado Líquido (R$)']:,.2f}")
                st.metric("Taxa de Aprovação", f"{best_strategy['Taxa de Aprovação (%)']:.1f}%")
                st.metric("Taxa de Inadimplência", f"{best_strategy['Taxa de Inadimplência (%)']:.1f}%")
            
            # Tabela detalhada
            st.write("### Detalhamento por Estratégia")
            
            # Formatar o DataFrame para exibição
            display_df = df_results.copy()
            
            # Formatar colunas numéricas
            money_cols = ['Valor da Carteira (R$)', 'Valor de Inadimplência (R$)', 'Receita de Juros (R$)', 'Resultado Líquido (R$)']
            percent_cols = ['Taxa de Aprovação (%)', 'Taxa de Inadimplência (%)', 'ROI (%)']
            
            for col in money_cols:
                display_df[col] = display_df[col].apply(lambda x: f"R$ {x:,.2f}")
                
            for col in percent_cols:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
                
            display_df['Solicitações Aprovadas'] = display_df['Solicitações Aprovadas'].apply(lambda x: f"{x:,.0f}")
            
            # Exibir tabela formatada
            st.dataframe(display_df, use_container_width=True)
            
            # Recomendações
            st.write("### Recomendações")
            
            recommendations = []
            
            # Identificar a estratégia com maior ROI e volume
            best_roi_strategy = df_results.loc[df_results['ROI (%)'].idxmax(), 'Estratégia']
            best_volume_strategy = df_results.loc[df_results['Solicitações Aprovadas'].idxmax(), 'Estratégia']
            
            if best_roi_strategy == best_volume_strategy:
                recommendations.append(f"- A estratégia **{best_roi_strategy}** oferece o melhor equilíbrio entre volume de aprovações e rentabilidade.")
            else:
                recommendations.append(f"- Para maximizar rentabilidade (ROI), a estratégia **{best_roi_strategy}** é a mais indicada.")
                recommendations.append(f"- Para maximizar volume de aprovações, a estratégia **{best_volume_strategy}** é a mais indicada.")
            
            # Avaliar estratégias específicas
            for i, row in df_results.iterrows():
                strategy_name = row['Estratégia']
                roi = row['ROI (%)']
                default_rate = row['Taxa de Inadimplência (%)']
                
                if roi < 0:
                    recommendations.append(f"- A estratégia **{strategy_name}** apresenta ROI negativo. É recomendável revisar os critérios de aprovação ou aumentar as taxas de juros.")
                
                if default_rate > 25:
                    recommendations.append(f"- A estratégia **{strategy_name}** apresenta taxa de inadimplência muito elevada (>{default_rate:.1f}%). Considere critérios mais rigorosos.")
            
            # Exibir recomendações
            for rec in recommendations:
                st.markdown(rec)
                
            # Adicionar opção de download dos resultados
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download dos Resultados (CSV)",
                data=csv,
                file_name="comparacao_estrategias.csv",
                mime="text/csv",
            )

    # Tab 3: Cenários Econômicos
    with portfolio_tab[2]:
        st.subheader("Simulação de Cenários Econômicos")
        st.write("Teste como diferentes cenários econômicos podem afetar a inadimplência e retorno de sua carteira.")
        
        # Configuração da carteira atual
        st.write("### Configuração da Carteira Base")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Valor da carteira
            portfolio_value = st.number_input(
                "Valor Total da Carteira (R$)",
                min_value=100000.0,
                max_value=100000000.0,
                value=10000000.0,
                step=1000000.0,
                format="%.2f"
            )
            
            # Taxa média de juros
            avg_interest_rate = st.slider(
                "Taxa Média de Juros (%)",
                min_value=5.0,
                max_value=50.0,
                value=25.0,
                step=0.5
            )
            
            # Taxa atual de inadimplência
            current_default_rate = st.slider(
                "Taxa Atual de Inadimplência (%)",
                min_value=1.0,
                max_value=30.0,
                value=8.0,
                step=0.5
            )
        
        with col2:
            # Distribuição da carteira por segmento de risco
            st.write("**Distribuição por Segmento de Risco:**")
            
            segment_low_risk = st.slider("Baixo Risco (%)", 0, 100, 30, 5)
            segment_medium_risk = st.slider("Médio Risco (%)", 0, 100, 40, 5)
            segment_high_risk = st.slider("Alto Risco (%)", 0, 100, 30, 5)
            
            # Verificar se a soma é 100%
            total_segments = segment_low_risk + segment_medium_risk + segment_high_risk
            if total_segments != 100:
                st.warning(f"A soma dos percentuais deve ser 100%. Atualmente: {total_segments}%")
        
        # Cenários econômicos
        st.write("### Cenários Econômicos")
        st.write("Configure diferentes cenários econômicos e seus impactos na inadimplência:")
        
        # Tabs para diferentes cenários
        scenario_tabs = st.tabs(["Cenário Base", "Otimista", "Pessimista", "Crise", "Personalizado"])
        
        # Definir cenários e seus impactos
        with scenario_tabs[0]:
            st.write("#### Cenário Base (Atual)")
            st.write("Mantém as condições econômicas atuais, sem mudanças significativas.")
            
            scenario_base = {
                'nome': 'Base',
                'descricao': 'Condições econômicas atuais',
                'multiplicador_baixo': 1.0,
                'multiplicador_medio': 1.0,
                'multiplicador_alto': 1.0,
                'variacao_pib': 1.5,
                'variacao_desemprego': 0.0,
                'variacao_juros': 0.0
            }
            
if __name__ == '__main__':
    # Executar a aplicação
    try:
        # Configuração já feita no início do arquivo
        # Remover qualquer outra configuração duplicada
        
        # Mostrar o módulo correspondente à seleção
        if app_mode == "Previsão de Fluxo de Caixa":
            # O código para este módulo já está definido acima
            pass  # Não precisamos adicionar nada aqui, o módulo já está definido
        # Os outros módulos continuam funcionando normalmente
    except Exception as e:
        st.error(f"Erro ao executar a aplicação: {e}")
        st.exception(e)
