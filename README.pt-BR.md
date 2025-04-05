# CreditWise

<div align="center">
  **Análise Avançada de Risco de Crédito com Explicabilidade ML**
  
  [![Python](https://img.shields.io/badge/Python-3.7%2B-4584b6?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![SHAP](https://img.shields.io/badge/SHAP-0.41.0-00C244?style=flat-square)](https://github.com/slundberg/shap)
  [![XGBoost](https://img.shields.io/badge/XGBoost-1.7.5-0073B7?style=flat-square)](https://xgboost.readthedocs.io/)
  [![LightGBM](https://img.shields.io/badge/LightGBM-3.3.5-3498DB?style=flat-square)](https://lightgbm.readthedocs.io/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
  [![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000?style=flat-square)](https://github.com/psf/black)
</div>

## Vídeo de Demonstração

Assista a uma demonstração da plataforma CreditWise em funcionamento:
[Clique aqui para ver o vídeo de demonstração](https://drive.google.com/file/d/12DYjNcVZ1p0uRPJLejAJil4P3jtXZQYE/view?usp=sharing)

## Visão Geral

O CreditWise é uma plataforma de análise de risco de crédito que utiliza machine learning e técnicas de explicabilidade para fornecer avaliações de crédito e insights acionáveis. A aplicação utiliza dados sintéticos para demonstrar técnicas de credit scoring e estratégias de cobrança.

### Capacidades Principais

1. **Geração e Processamento de Dados** - Cria perfis sintéticos de clientes com atributos financeiros e históricos de pagamento
2. **Implementação de Modelos ML** - Utiliza modelos ensemble (Random Forest, XGBoost, LightGBM, MLP, LogisticRegression) para previsão de inadimplência
3. **IA Explicável** - Usa SHAP para fornecer decisões de modelo transparentes com explicações globais e locais
4. **Simulação de Portfólio** - Testa estratégias de aprovação e cenários econômicos para otimizar métricas de risco-retorno
5. **Análise de Clientes** - Entrega avaliação individual de risco com detalhamento de fatores
6. **Simulação de Cobrança** - Otimiza estratégias individuais de recuperação de crédito através de abordagens baseadas em risco

## Estrutura Atual do Projeto

```
creditwise/
├── app.py                 # Aplicação principal Streamlit
├── requirements.txt       # Dependências Python
├── README.md              # Documentação em inglês
├── README.pt-BR.md        # Documentação em português
└── creditwise_env/        # Ambiente virtual Python
```

## Detalhes de Implementação

### Gestão de Dados

- Perfis sintéticos de clientes com mais de 10 atributos financeiros (idade, renda, tempo de emprego, etc.)
- Geração de dados sintéticos de cobrança para clientes inadimplentes
- Pré-processamento incluindo escalonamento e divisão treino-teste

### Motor Analítico

- Múltiplas implementações de modelos (Random Forest, XGBoost, LightGBM, MLP, LogisticRegression)
- Métricas de desempenho: ROC-AUC, precision-recall, F1 score
- Validação cruzada para avaliação de estabilidade do modelo
- Análise e classificação de importância de features

### Interface Interativa

- Visualização da distribuição de risco do portfólio
- Dashboards de desempenho do modelo com métricas detalhadas
- Ferramentas de análise exploratória de dados
- Análise de correlação e distribuições de variáveis

### Explicabilidade de Decisões

- Visualização de importância global de features
- Gráficos de dependência SHAP para análise de relações entre features
- Explicações de previsões individuais para decisões transparentes
- Insights técnicos traduzidos em recomendações acionáveis para negócios

### Análise de Risco

- Teste de limiar de aprovação com análise de impacto em métricas de negócios
- Análise de desempenho do portfólio
- Visualização do trade-off risco-retorno
- Comparação de modelos com métricas de desempenho

### Estratégias de Cobrança

- Análise de desempenho de cobrança
- Comparação de efetividade entre canais
- Análise do tempo de recuperação
- Cálculos de custo-benefício para esforços de cobrança

## Módulos da Aplicação

### Visão Geral
- Introdução ao credit scoring
- Visualização da distribuição de scores
- Métricas-chave de desempenho

### Exploração de Dados
- Estatísticas descritivas
- Distribuições de variáveis
- Análise de correlação
- Análise bivariada

### Modelo de Crédito
- Métricas de desempenho do modelo
- Análise de curva ROC
- Visualização de importância de features
- Análise de limiar de decisão

### Comparação de Modelos
- Métricas de desempenho entre múltiplos algoritmos
- Resultados de validação cruzada
- Comparação de matrizes de confusão

### Explicabilidade (SHAP)
- Importância global de features
- Gráficos de dependência de features
- Explicações de decisões locais
- Análise individual de cliente

### Simulador de Crédito
- Criação de perfil de cliente
- Previsão de score de crédito
- Análise de fatores de decisão
- Testes de cenários "e se"

### Análise de Cobrança
- Perfis de clientes inadimplentes
- Avaliação de estratégias de cobrança
- Simulação de recuperação
- Análise de ROI para esforços de cobrança

## Benchmarks de Desempenho

| Modelo | ROC-AUC | Precisão | Recall | F1 Score |
|--------|---------|----------|--------|----------|
| Random Forest | 0,82 | 0,76 | 0,71 | 0,73 |
| XGBoost | 0,85 | 0,79 | 0,73 | 0,76 |
| LightGBM | 0,84 | 0,78 | 0,72 | 0,75 |
| MLP | 0,81 | 0,74 | 0,70 | 0,72 |
| Logistic Regression | 0,79 | 0,73 | 0,69 | 0,71 |

### Fatores de Risco Primários

Os modelos identificam estes preditores-chave para risco de inadimplência de crédito:
1. Histórico de pagamentos (padrões de inadimplência)
2. Taxa de utilização de crédito
3. Relação dívida-renda
4. Estabilidade de emprego
5. Frequência de consultas de crédito

## Stack Tecnológica

- **Core**: Python
- **Processamento de Dados**: Pandas, NumPy
- **Framework ML**: Scikit-learn, XGBoost, LightGBM
- **Explicabilidade**: SHAP
- **Frontend**: Streamlit
- **Visualização**: Plotly, Matplotlib, Seaborn

## Instalação

### Requisitos

- Python 3.7+
- Gerenciador de pacotes pip
- Ambiente virtual (recomendado)

### Configuração

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/creditwise.git
cd creditwise

# Criar e ativar ambiente virtual
python -m venv creditwise_env
source creditwise_env/bin/activate  # No Windows: creditwise_env\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Iniciar aplicação
streamlit run app.py
```

A aplicação estará acessível em http://localhost:8501

## Contribuição

Recebemos contribuições que aprimorem as capacidades do CreditWise:

1. Faça um Fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/aprimoramento`)
3. Faça commit das alterações (`git commit -m 'Adiciona aprimoramento'`)
4. Faça push para a branch (`git push origin feature/aprimoramento`)
5. Abra um Pull Request

## Licença

Liberado sob Licença MIT. Veja `LICENSE` para detalhes.


---

<div align="center">
  <p>
    <b>CreditWise</b> - Análise de Risco de Crédito com IA Explicável
  </p>
</div> 
