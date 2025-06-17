# 🫀 Classificação Binária: Doença Cardíaca Isquêmica (IHD)

Este projeto implementa dois modelos de aprendizado de máquina, **Naive Bayes** e **MLP**, para realizar **classificação binária** da presença ou ausência de **Doença Cardíaca Isquêmica (IHD)** com base em variáveis clínicas e de estilo de vida.

---

## Dataset

- **Fonte:** [Kaggle - Quantum Enhanced Ischemic Heart Disease Dataset](https://www.kaggle.com/datasets/ziya07/quantum-enhanced-ischemic-heart-disease-dataset)
- **Alvo:** `IHD` → 0 (ausência) ou 1 (presença) de isquemia cardíaca
- **Atributos utilizados:**
  - Idade
  - Gênero
  - Pressão Sistólica
  - Colesterol
  - IMC
  - Tabagismo
  - Atividade Física
  - Diabetes
  - Hipertensão

---

## Modelos Utilizados

### Naive Bayes
- Algoritmo probabilístico (GaussianNB)
- Assumindo independência entre as variáveis

### MLP (Multi-Layer Perceptron)
- Arquitetura:
  - 2 camadas ocultas: 16 e 8 neurônios
  - Função de ativação: ReLU
  - Otimizador: Adam
  - Iterações: até 10.000 épocas

---

## Como executar

git clone https://github.com/seu-usuario/classificacao-binaria.git
cd classificacao-binaria
python classificacao_ihd.py
