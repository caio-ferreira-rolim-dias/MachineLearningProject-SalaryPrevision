# tratamento dos dados
import numpy as np
import pandas as pd

# graficos
import seaborn as sns
import matplotlib.pyplot as plt

# auxiliando na plotagem da matriz de confusao
import matplotlib

# separando os dados 
from sklearn.model_selection import train_test_split, learning_curve,KFold, StratifiedKFold, LeaveOneOut, cross_validate, validation_curve

# feature engenieering
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# modelagem
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# metricas de eficiencia do modelo
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, recall_score, precision_score, classification_report, confusion_matrix

# fazer o cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# auxiliar para aleatoriedade
import random


"""
O objetivo desse projeto era criar um modelo de classificação, que, com base nos dadaset fornecido, nos permitisse prever quem reveberia maide de 50K dolares e quem receberia até 50K dolares no ano posterior. Utilizei DecisionTree para o modelo(foi uma opção para fins didáticos), em seguida realizei a validação cruzada para buscar escolher o melhor parâmetro de ```ccp_alpha```. Hiperparâmetro este que escolhi para controlar o pruning do meu modelo.
"""

#Realizando a importação dos dados
df_salario = pd.read_csv('https://raw.githubusercontent.com/abnr/ml-data/main/census.csv')

#Não existiam registros duplicados no dataset


# verificando a quantidade de missing values por coluna
df_salario.isna().sum()

# limpando espaços indejadados antes e após as strings de todas as features categoricas
df_salario['workclass'] = df_salario['workclass'].str.strip()
df_salario['education'] = df_salario['education'].str.strip()
df_salario['maritalstatus'] = df_salario['maritalstatus'].str.strip()
df_salario['occupation'] = df_salario['occupation'].str.strip()
df_salario['relationship'] = df_salario['relationship'].str.strip()
df_salario['race'] = df_salario['race'].str.strip()
df_salario['sex'] = df_salario['sex'].str.strip()
df_salario['nativecountry'] = df_salario['nativecountry'].str.strip()
df_salario['over50k'] = df_salario['over50k'].str.strip()

# verifiquei que em algumas features haviam dados inconsistentes, optei por elimina-los. Mas antes verifiquei o impacto dessas eliminações.
# verificar o percentual que esses dados inconsistentes correspondem do total da feature
((df_salario['workclass'].value_counts())/df_salario['workclass'].value_counts().sum())*100

# removendo os valores inconsistentes. Optei por fazer isso, e continuar com a coluna, por acho que é uma feature relevante e os
#...valores inconsistents corresponder a apenas 5.6% dela.
# usei a função loc para criar um dataframe com todos os valores inconsistentes
workclass_remove = df_salario.loc[df_salario['workclass'] == '?']

# estou dropando apenas os valores inconsistentes que havia armazena no dataframe workclass_remove
df_salario_prep = df_salario.drop(workclass_remove.index)

# verificando agora a quantidade de valores inconsistentes da feature 'occupation' restaram apenas 0.023%. Vou dropalos também
((df_salario_prep['occupation'].value_counts())/df_salario_prep['occupation'].value_counts().sum())*100
occupation_remove = df_salario_prep.loc[df_salario_prep['occupation'] == '?']
df_salario_prep = df_salario_prep.drop(occupation_remove.index)

# trocando os valores da variável target por 0 e 1
over50k_dict = {'<=50K': 0, '>50K': 1}
df_salario_prep['over50k'] = df_salario_prep['over50k'].map(over50k_dict)

# criando dois DataFrame com os dados preparados. Um com as preditoras e outro com a target
X_df_salario_prep = df_salario_prep.drop('over50k', axis=1)
y_df_salario_prep = pd.DataFrame(df_salario_prep['over50k'], columns=['over50k'])

# feature engineering
X_train_numeric = X_df_salario_prep.select_dtypes(exclude='object')
X_train_categoric = X_df_salario_prep.select_dtypes(include='object')
# tratando parte das variaveis categoricas com OneHotEncoder
cat_cols = ['workclass', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex']
X_train_categoric_onehot = X_train_categoric.loc[:, cat_cols]
ohe = OneHotEncoder(sparse=False)
X_train_categoric_onehot = ohe.fit_transform(X_train_categoric_onehot)
X_train_categoric_onehot = pd.DataFrame(X_train_categoric_onehot, columns=ohe.get_feature_names())
# tratando as demais variaveis categoricas com OrdinalEncoder
X_train_categoric_class = X_train_categoric.loc[:, ['education']]
# abaixo criei uma lista para dar um peso para cada valor, a ordem abaixo vai ser a ordem de valoraçaõ que o OrdinalEncoder vai dar
order_levels = ['Doctorate', 'Masters', 'Bachelors', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Prof-school', 'HS-grad', 
               '12th', '11th', '10th', '9th', '7th-8th', '5th-6th', '1st-4th', 'Preschool']
oe = OrdinalEncoder(categories=[order_levels])
X_train_categoric_oe = oe.fit_transform(X_train_categoric_class)
X_train_categoric_oe = pd.DataFrame(X_train_categoric_oe, columns=['education'])

# abaixo estou concatenando os o df_categoric que tratei com o OneHotEncoder, com o que tratei com o OndinalEncoder, com o 
#...df_numeric que foi sepadado no inicio doalgoritmo, mas não precisou de tratamento.
#...ao tratar o df_categoric o codigo fez automaticamente um reset_index, precisei no codigo abaixo fazer a mesma coisa com o
#...df_numeric senão eu teria coluna divergencia nos index o que criaria novas rows.
X_df_salario_final = pd.concat([X_train_categoric_oe, X_train_categoric_onehot, X_train_numeric.reset_index(drop=True)], axis=1)

# dividindo os dados em train e test
X_train, X_test, y_train, y_test = train_test_split(X_df_salario_final, y_df_salario_prep, train_size=0.8, random_state=123)

# construindo o modelo
clf = DecisionTreeClassifier(criterion='gini')
# treinando o modelo
clf.fit(X_train, y_train)
# rodando o modelo
y_pred = clf.predict(X_test)

# analisando o modelo criado através de uma confusion matrix e métricas como accuracy, precisione  recall
# abaixo criei uma formula para otimizar a apresentacao da matriz de confusao
def plot_confusion_matrix(y_test, y_pred):
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True,
                cbar=False,
                cmap= matplotlib.cm.get_cmap('gist_yarg'),
                fmt='.0f') #tira a notação científica
    plt.ylabel('Real')
    plt.xlabel('Predicted')
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
plot_confusion_matrix(y_test, y_pred)

print(f'A acurácia do treino foi: {accuracy_score(y_train, clf.predict(X_train))}')
print(f'A acurácia do teste foi: {accuracy_score(y_test, clf.predict(X_test))}')

# dos valores que eu previ como 1 (receberam mais de 50K) quanto porcento realmente receberam acima
print(f'A precisão do treino foi: {precision_score(y_train, clf.predict(X_train))}')
print(f'A precisão do teste foi: {precision_score(y_test, clf.predict(X_test))}')

# dos valores que eram 1 (receberam mais de 50K) quantos porcento(%) eu consegui prever
print(f'O recall do treino foi: {recall_score(y_train, clf.predict(X_train))}')
print(f'O recall do teste foi: {recall_score(y_test, clf.predict(X_test))}')

# cross validation
cv_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

param_range = list(np.arange(0, 0.02, 0.001))
param_name = 'ccp_alpha'

def cross_validation(estimator=None, X=None, y=None, param_name=None, param_range=None, scoring=None, cv=None):
    train_score, val_score = validation_curve(estimator=estimator,
                                             X=X,
                                             y=y,
                                             param_name=param_name,
                                             param_range=param_range,
                                             scoring=scoring,
                                             n_jobs=-1,
                                             cv=cv)
    plt.figure(figsize=(8,6))
    #plotando as curvas de teste e validação
    plt.plot(param_range, np.mean(train_score, 1), color= 'blue', label= 'training score')
    plt.plot(param_range, np.mean(val_score, 1), color= 'red', label= 'validation score')
    plt.legend(loc='best')
    plt.xlabel('Param_range')
    plt.ylabel('Score')
    best_param = param_range[np.argmax(np.mean(val_score, 1))]
    return best_param

best_param_kfold_stratified = cross_validation(estimator=clf,
                                               X=X_train,
                                               y=y_train,
                                               param_name=param_name,
                                               param_range=param_range,
                                               scoring='accuracy',
                                               cv=cv_kfold)
print(f'O melhor ccp_alpha para KFold estratificado é: {best_param_kfold_stratified}')

# aplicando o parametro otimizado
clf_otimizado = tree.DecisionTreeClassifier(criterion='gini', ccp_alpha=0.001)
clf_otimizado.fit(X_train, y_train)

y_pred_otimizado = clf_otimizado.predict(X_test)
plot_confusion_matrix(y_test, y_pred_otimizado)

print(f'A acurácia do treino foi: {accuracy_score(y_train, clf_otimizado.predict(X_train))}')
print(f'A acurácia do teste foi: {accuracy_score(y_test, clf_otimizado.predict(X_test))}')

# dos valores que eu previ como 1 (receberam mais de 50K) quanto porcento realmente receberam acima
print(f'A precisão do treino foi: {precision_score(y_train, clf_otimizado.predict(X_train))}')
print(f'A precisão do teste foi: {precision_score(y_test, clf_otimizado.predict(X_test))}')

# dos valores que eram 1 (receberam mais de 50K) quantos porcento(%) eu consegui prever

print(f'O recall do treino foi: {recall_score(y_train, clf_otimizado.predict(X_train))}')
print(f'O recall do teste foi: {recall_score(y_test, clf_otimizado.predict(X_test))}')

# Validation curves do modelo otimizado
validation_curve(clf, X_train)
cross_validation(estimator=clf_otimizado,
                 X=X_train,
                 y=y_train,
                 param_name=param_name,
                 param_range=param_range,
                 scoring='accuracy',
                 cv=cv_kfold)