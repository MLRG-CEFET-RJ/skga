import pandas as pd
from scipy.stats import f_oneway

lista_arquivos = [
    [
        "DecisionTreeClassifier_GridSearchCV_cred.csv",
        "DecisionTreeClassifier_HyperBRKGASearchCV_cred.csv",
        "DecisionTreeClassifier_RandomizedSearchCV_cred.csv"
    ],
    [
        "DecisionTreeRegressor_GridSearchCV_diamond.csv",
        "DecisionTreeRegressor_HyperBRKGASearchCV_diamond.csv",
        "DecisionTreeRegressor_RandomizedSearchCV_diamond.csv"
    ],
    [
        "KNeighborsClassifier_GridSearchCV_cred.csv",
        "KNeighborsClassifier_HyperBRKGASearchCV_cred.csv",
        "KNeighborsClassifier_RandomizedSearchCV_cred.csv"
    ],
    [
        "KNeighborsRegressor_GridSearchCV_diamond.csv",
        "KNeighborsRegressor_HyperBRKGASearchCV_diamond.csv",
        "KNeighborsRegressor_RandomizedSearchCV_diamond.csv"
    ],
    [
        "Lasso_GridSearchCV_diamond.csv",
        "Lasso_HyperBRKGASearchCV_diamond.csv",
        "Lasso_RandomizedSearchCV_diamond.csv"
    ],
    [
        "LinearRegression_GridSearchCV_diamond.csv",
        "LinearRegression_HyperBRKGASearchCV_diamond.csv",
        "LinearRegression_RandomizedSearchCV_diamond.csv"
    ],
    [
        "LogisticRegression_GridSearchCV_cred.csv",
        "LogisticRegression_HyperBRKGASearchCV_cred.csv",
        "LogisticRegression_RandomizedSearchCV_cred.csv"
    ],
    [
        "MLPClassifier_GridSearchCV_cred.csv",
        "MLPClassifier_HyperBRKGASearchCV_cred.csv",
        "MLPClassifier_RandomizedSearchCV_cred.csv"
    ],
    [
        "MLPRegressor_GridSearchCV_diamond.csv",
        "MLPRegressor_HyperBRKGASearchCV_diamond.csv",
        "MLPRegressor_RandomizedSearchCV_diamond.csv"
    ],
    [
        "RandomForestClassifier_GridSearchCV_cred.csv",
        "RandomForestClassifier_HyperBRKGASearchCV_cred.csv",
        "RandomForestClassifier_RandomizedSearchCV_cred.csv"
    ],
    [
        "XGBClassifier_GridSearchCV_cred.csv",
        "XGBClassifier_HyperBRKGASearchCV_cred.csv",
        "XGBClassifier_RandomizedSearchCV_cred.csv"
    ]
]

resultados = []
alpha = .05

for i in range(len(lista_arquivos)):
    amostras = []

    for j in range(0, 3):
        arquivo = lista_arquivos[i][j]
        dados = pd.read_csv(arquivo, header=0)
        dados = dados[:-2]

        amostra = dados["Score"]
        amostras.append(amostra)

    statistic, p_value = f_oneway(*amostras)
    resultados.append((statistic, p_value))

with open("resultados_anova.txt", "w") as file:
    for i, resultado in enumerate(resultados):
        estimador = lista_arquivos[i][0].split("_")[0]
        statistic, p_value = resultado

        file.write(f"{estimador}:\n")
        file.write(f"Estatística do teste = {statistic:.4f} | Valor p = {p_value}\n")
        file.write(f"Conclusão: \n")
        if p_value < alpha:
            file.write("Existe uma diferença significativa entre pelo menos um par de grupos\n\n")
        else:
            file.write("As médias não são significativamente diferentes\n\n")

        file.write("-----------------------------------------------\n\n")


print("Os resultados foram salvos no arquivo resultados_anova.txt")