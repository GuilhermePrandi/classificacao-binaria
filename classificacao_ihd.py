import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# Carregamento dos dados
data = pd.read_csv("ihd_dataset.csv")

# Separar variáveis
X = data.drop("IHD", axis=1)
y = data["IHD"]

# Definir colunas categóricas e numéricas
categorical_cols = ["Gender", "Smoking_Status", "Physical_Activity"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Normalização dos dados
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Separar em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Pipeline para Naive Bayes
nb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GaussianNB())
])
nb_pipeline.fit(X_train, y_train)
y_nb_pred = nb_pipeline.predict(X_test)

# Pipeline para MLP
mlp_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        max_iter=10000,
        random_state=42,
        verbose=True
    ))
])
mlp_pipeline.fit(X_train, y_train)
y_mlp_pred = mlp_pipeline.predict(X_test)

# Matriz de Confusão
print("Matriz de Confusão - Naive Bayes:")
print(confusion_matrix(y_test, y_nb_pred))
print("\nMatriz de Confusão - MLP:")
print(confusion_matrix(y_test, y_mlp_pred))

# F1-Score
f1_nb = f1_score(y_test, y_nb_pred)
f1_mlp = f1_score(y_test, y_mlp_pred)

print("\nRelatório Naive Bayes:")
print(classification_report(y_test, y_nb_pred))
print("\nRelatório MLP:")
print(classification_report(y_test, y_mlp_pred))

# Tabela final comparativa entre Naive Bayes e MLP
print("\nTabela Comparativa de F1-Score:")
print(f"{'Modelo':<15} {'F1-Score':<10}")
print(f"{'Naive Bayes':<15} {f1_nb:.4f}")
print(f"{'MLP':<15} {f1_mlp:.4f}")
