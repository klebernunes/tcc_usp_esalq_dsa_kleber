# -*- coding: utf-8 -*-

# MBA USP ESALQ
#
# TCC: Classificação de pacientes para direcionamento de programas de saúde preventivos
# Kleber S. Nunes
# 
# Orientador: Thiago Gentil Ramires
# script adaptado 
# Prof. Dr. Helder Prado
# Prof. Dr. Wilson Tarantin Jr.
# 
# Processamento Modelos
#
#%% Importando os pacotes necessários

import pandas                        as     pd
import numpy                         as     np
import matplotlib.pyplot             as     plt
import seaborn                       as     sns

from   sklearn.tree                  import DecisionTreeClassifier
from   sklearn.model_selection       import train_test_split
from   sklearn.ensemble              import RandomForestClassifier
from   sklearn.linear_model          import LogisticRegression
from   sklearn.metrics               import accuracy_score
from   sklearn.ensemble              import BaggingClassifier
from   sklearn.metrics               import confusion_matrix, ConfusionMatrixDisplay
from   sklearn.metrics               import classification_report
from   sklearn.ensemble              import AdaBoostClassifier
from   sklearn.ensemble              import GradientBoostingClassifier
from   sklearn.model_selection       import GridSearchCV
from   yellowbrick.classifier.rocauc import roc_auc
from   pandas_profiling              import ProfileReport 

#%% Carregando o dataset
print("------------------ início fase preparação --------------------")
df = pd.read_csv("base_pesquisa_jan_2022_a_dez_2022.csv",delimiter=";")

#%% realizando analise do dataset antes de qualquer manipulação
# Características das variáveis do dataset
df.info()

#%% Realizando o processo de inputação de dados
#-----------------------------------------------------------------
# troca campos nulos para zero, os mesmos só existem nas colunas de 
# ind_atd_ambulatorial, ind_tipo_internacao, ind_regime_internacao 
# ind_atd_ambulatorial é mutuamente exclusivo em relação 
# aos atributos ind_tipo_internacao e ind_regime_internacao
# assim ambos ganham a categoria zero
#-----------------------------------------------------------------
df_base = df.fillna(value = 0)
#%% Ajustando tipos de dados 
#-----------------------------------------------------------------
# inicialmente todos os dados são transformados para categóricos
# já que a maioria tem esse tipo 
#-----------------------------------------------------------------
df_ajustado = df_base.astype("category")
del df_base
#-----------------------------------------------------------------
# ajusta os campos qtd_realizada e idade para númericos
#-----------------------------------------------------------------
df_ajustado["QTD_REALIZADA"]= df_ajustado["QTD_REALIZADA"].astype("int64")
df_ajustado["IDADE"]= df_ajustado["IDADE"].astype("int64")

#%% realizando analise do dataset após 
#Características das variáveis do dataset
df_ajustado.info()

#Estatísticas univariadas
df_ajustado.describe()
resultado = df_ajustado.describe()
print(resultado)
del resultado

#%% Analisando o dataset
profile_ajustado = ProfileReport(df_ajustado, minimal=True)
profile_ajustado.to_file(output_file="overview_base_dados.html")
del profile_ajustado


#%% realizaçào do processo de seleção dos dados de interesse

df = df_ajustado

df = df.loc[~df['CATEGORIA_CID'].str.contains("Z")]
df = df.loc[~df['CAPITULO_CID'].isin(['16','17','18','19','20','21','22'])]
df = df.loc[df['COD_CLASSIFICACAO_PAI'].isin(['HON','DIG'])]
df = df.loc[df['CAPITULO_CID'].isin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])]
df = df[df['MES_REALIZACAO'].isin([7,8,9,10,11,12])]


#%% removendo atributos utilizados no processo de seleção ou desnecessários como variaveis explicativas

df = df.drop('COD_CLASSIFICACAO_PAI', axis=1)
df = df.drop('COD_CLASSIFICACAO',     axis=1)
df = df.drop('QTD_REALIZADA',         axis=1)
df = df.drop('CATEGORIA_CID',         axis=1)
df = df.drop('MES_REALIZACAO',        axis=1)

#%% verificando perfil final da base

profile_reduzido = ProfileReport(df, minimal=True)
profile_reduzido.to_file(output_file="overview_base_dados_final.html")
del profile_reduzido

#%% Transformando variáveis categóricas em dummies
df = pd.get_dummies(df, columns=['COD_PROCEDIMENTO','IND_SEXO','IND_TIPO_ATD_AMBULATORIAL','IND_REGIME_INTERNACAO','IND_TIPO_INTERNACAO'])
del df_ajustado
print("------------------ final fase preparação --------------------")
#%% Separando as variáveis Y e X
print("------------------ início fase modelagem --------------------")
X = df.drop(columns=['CAPITULO_CID']).values
y = df['CAPITULO_CID'].values 

#%% Coletar os nomes das variáveis X

features = list(df.drop(columns=['CAPITULO_CID']).columns)


#%% Criando amostras de treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Bagging
#%% 1: Para fins de comparação, estima-se uma árvore de classificação

tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf.fit(X_train, y_train)

# Predict do modelo de uma árvore
y_pred_tree = tree_clf.predict(X_test)

# Matriz de classificação para uma árvore
print(classification_report(y_test, y_pred_tree))


#%%  1a matriz de confusão
plt.figure(figsize=(10,6))

fx=sns.heatmap(confusion_matrix(y_test,y_pred_tree), annot=True, fmt=".2f",cmap="viridis")
fx.set_title('Confusion Matrix DecisionTreeClassifier uma árvore\n');
fx.set_xlabel('\n Predicted Values\n')
fx.set_ylabel('Actual Values\n');
plt.show()

#%% 1b curva ROC 

classes=y
roc_auc( 
          tree_clf,
          X_train, y_train, X_test, y_test,
          per_class=False
          )
plt.tight_layout()

#%% 1c limpando variaveis desnecessarias para os próximos processamentos
del tree_clf
del y_pred_tree



#%% 2: Estimando um modelo bagging com base em árvores de classificação

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_depth=4), # modelo base: árvore de classificação
    n_estimators=500,
    max_samples=50,
    bootstrap=True, # bootstrap = True indica modelo Bagging / False = Pasting
    n_jobs=-1, # utiliza todos os núcleos do computador
    random_state=42) 

bag_clf.fit(X_train, y_train)

# Predict do modelo bagging de árvores
y_pred_bag = bag_clf.predict(X_test)

# Gerando a matriz de confusão

cm = confusion_matrix(y_test, 
                      y_pred_bag, 
                      labels=bag_clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=bag_clf.classes_)

# Matriz de classificação do modelo bagging de árvores
print(classification_report(y_test, y_pred_bag))


#%%parte 2a matriz de confusão
plt.figure(figsize=(10,6))

fx=sns.heatmap(confusion_matrix(y_test,y_pred_bag), annot=True, fmt=".2f",cmap="viridis")
fx.set_title('Confusion Matrix BaggingClassifier DecisionTreeClassifier\n');
fx.set_xlabel('\n Predicted Values\n')
fx.set_ylabel('Actual Values\n');
plt.show()

#%% 2b curva ROC

classes=y
roc_auc( 
          bag_clf,
          X_train, y_train, X_test, y_test,
          per_class=False
          )
plt.tight_layout()

#%% 2c limpando variaveis desnecessarias para os próximos processamentos
del bag_clf
del y_pred_bag

#%% 3: Avaliação out-of-bag

# As observações de treinamento que não são amostradas são "out-of-bag"
# O modelo pode ser avaliado nessas observações sem a necessidade de um conjunto de validação
# Trata-se de uma avaliação automática após o treinamento

bag_clf_oob = BaggingClassifier(
    DecisionTreeClassifier(max_depth=4), 
    n_estimators=500,  
    max_samples=50,
    bootstrap=True,
    n_jobs=-1, 
    oob_score=True, # avaliação out-of-bag
    random_state=42) 

bag_clf_oob.fit(X, y)

# Acurácia do modelo
print(bag_clf_oob.oob_score_)
#%% 3a limpando variaveis desnecessarias para os próximos processamentos
del bag_clf_oob


#%% 4: Estimando um modelo bagging com base em uma logística

bag_log = BaggingClassifier(
    LogisticRegression(), # modelo base: logística
    n_estimators=300, # 500
    max_samples=50,
    bootstrap=True, # bootstrap = True indica modelo Bagging / False = Pasting
    n_jobs=-1, # utiliza todos os núcleos do computador
    random_state=42) 

bag_log.fit(np.delete(X_train, -1, axis=1), y_train)

# Predict do modelo bagging de logística
y_pred_log = bag_log.predict(np.delete(X_test, -1, axis=1))

# Matriz de classificação do modelo bagging de logística
print(classification_report(y_test, y_pred_log))

#%%parte 4a  # matriz de confusão
plt.figure(figsize=(10,6))

fx=sns.heatmap(confusion_matrix(y_test,y_pred_log), annot=True, fmt=".2f",cmap="viridis")
fx.set_title('Confusion Matrix BaggingClassifier DecisionTreeClassifier\n');
fx.set_xlabel('\n Predicted Values\n')
fx.set_ylabel('Actual Values\n');
plt.show()

#%% 4b Curva ROC

classes=y
roc_auc( 
          bag_log,
          np.delete(X_train, -1, axis=1), y_train, np.delete(X_test, -1, axis=1), y_test,
          per_class=False
          )
plt.tight_layout()

#%% 4c limpando variaveis desnecessarias para os próximos processamentos
del bag_log
del y_pred_log

#%% 5: Random Forests

# O ForestClassifier é mais otimizado para árvores de decisão

rnd_clf = RandomForestClassifier(n_estimators=500, max_depth=5, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

# Predict na base de teste
y_pred_rf = rnd_clf.predict(X_test)

# Matriz de classificação
print(classification_report(y_test, y_pred_rf))
#%% 5a
# Importância das variáveis X
for name, score in zip(features, rnd_clf.feature_importances_):
   print(name, score)
   
#%% 5b matriz de confusão
# matriz de confusão
plt.figure(figsize=(10,6))

fx=sns.heatmap(confusion_matrix(y_test,y_pred_rf), annot=True, fmt=".2f",cmap="viridis")
fx.set_title('Confusion Matrix RandomForestClassifier\n');
fx.set_xlabel('\n Predicted Values\n')
fx.set_ylabel('Actual Values\n');
plt.show()   
#%% 5c curva ROC

roc_auc(  rnd_clf,
             X_train, y_train, X_test, y_test,
             per_class=False
             )

plt.tight_layout()
#%% 5d limpando variaveis desnecessarias para os próximos processamentos
del rnd_clf
del y_pred_rf

#%% Boosting
#%% 6: AdaBoost

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=500,
    algorithm='SAMME.R',
    learning_rate=0.1)

ada_clf.fit(X_train,y_train)

# Predict na base de teste
y_pred_ada = ada_clf.predict(X_test)

# Matriz de classificação
print(classification_report(y_test, y_pred_ada))

#%% 6a: Grid search para escolha da profundidade 

gs_ada = GridSearchCV(estimator=DecisionTreeClassifier(),
                          param_grid={
                              'max_depth': [2, 3, 4, 5, 6, 7 , 8, 9, 10, None],
                          },
                          cv=5,
                          return_train_score=False,
                          scoring='accuracy')

gs_ada.fit(X=X_train, y=y_train)

resultados_gs_ada = pd.DataFrame(gs_ada.cv_results_).set_index('rank_test_score').sort_index()

print(resultados_gs_ada)
# melhores parâmetros
print(gs_ada.best_params_)   

#%% 6b: Lista com cada iteração

estimators = np.arange(1,501)

#%% 6c: Lista que vai receber cada resultado das iterações

scores_train = np.zeros(500, dtype=np.float64)
scores_test = np.zeros(500, dtype=np.float64)

#%% 6d: Coletando a acurácia de cada iteração nos dados de treino

for i, y_pred in enumerate(ada_clf.staged_predict(X_train)): 
    acc = accuracy_score(y_train, y_pred) 
    scores_train[i] = acc 
#print(scores_train)
    
#%% 6e: Coletando a acurácia de cada iteração nos dados de teste

for i, y_pred in enumerate(ada_clf.staged_predict(X_test)):
    acc = accuracy_score(y_test, y_pred)
    scores_test[i] = acc
    
#print(scores_test)
     
#%% 6f: Visualizando a acurácia ao longo de cada iteração

plt.figure(figsize=(12, 10))
plt.title("Acc por iteração")
plt.plot(estimators,scores_train, label='Dados de treino')
plt.plot(estimators,scores_test, label='Dados de teste')
plt.legend(loc="upper right")
plt.xlabel("Iterations", fontsize=16)
plt.ylabel("Acurácia", fontsize=16)
plt.show()

#%% 6g: Parametrizando o AdaBoost com base nas análise anteriores

ada_clf_best = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=None),
    n_estimators=340,
    algorithm='SAMME.R',
    learning_rate=0.1)

ada_clf_best.fit(X_train,y_train)

# Predict na base de teste
y_pred_ada_best = ada_clf_best.predict(X_test)

# Matriz de classificação
print(classification_report(y_test, y_pred_ada))
#%% 6h Matriz de confusão 

# matriz de confusão
plt.figure(figsize=(10,6))

fx=sns.heatmap(confusion_matrix(y_test,y_pred_ada_best), annot=True, fmt=".2f",cmap="viridis")
fx.set_title('Confusion Matrix AdaBoostClassifier\n');
fx.set_xlabel('\n Predicted Values\n')
fx.set_ylabel('Actual Values\n');
plt.show()
#%% 6i Curva ROC

roc_auc( 
             ada_clf_best,
             X_train, y_train, X_test, y_test,
             per_class=False
             )
plt.tight_layout()
#%% 6j limpando variaveis desnecessarias para os próximos processamentos
del ada_clf
del y_pred_ada
del ada_clf_best
del y_pred_ada_best

#%% 7 Gradiente Boosting

gbc_cls = GradientBoostingClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42)
gbc_cls.fit(X_train, y_train)

y_pred_gbc = gbc_cls.predict(X_test)

# Matriz de classificação
print(classification_report(y_test, y_pred_gbc))
#%% 7a matriz de confusão
# matriz de confusão
plt.figure(figsize=(10,6))

fx=sns.heatmap(confusion_matrix(y_test,y_pred_gbc), annot=True, fmt=".2f",cmap="viridis")
fx.set_title('Confusion Matrix AdaBoostClassifier\n');
fx.set_xlabel('\n Predicted Values\n')
fx.set_ylabel('Actual Values\n');
plt.show()
   
#%% 7b Curva ROC

roc_auc( 
             gbc_cls,
             X_train, y_train, X_test, y_test,
             per_class=False
             )
plt.tight_layout()

#%% 6j limpando variaveis desnecessarias para os próximos processamentos
del gbc_cls
del y_pred_gbc

#%%
print("------------------ final da modelagem --------------------")
#%% FIM!
