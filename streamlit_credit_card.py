# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:04:21 2023

@author: Y.J.T.N
"""

import os
import pandas as pd
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
#from sklearn.model_selection import GridSearchCV

# classifeir Librairies
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
import collections


# other Librairies
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalance_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("data/creditcard.csv")

st.sidebar.title("Credit Card Fraud Detection")

pages = ["project Context", "Data Exploratory", "Data Analysis" ,"Data Preprocessing", "Modeling", "Conclusion"]


page = st.sidebar.radio("Select a page", pages)

if page == pages[0]:

    st.write("## Project Context")
    st.write("La fraude bancaire représente chaque année des pertes considérables pour les institutions financières.")
    st.write("Les méthodes traditionnelles basées sur des règles statiques sont souvent insuffisantes face à l’évolution rapide des techniques de fraude.")
    st.write("C’est pourquoi les banques s’appuient désormais sur des approches data-driven et apprenantes pour améliorer la détection en temps réel.")
    st.image("img/image_projet.png")

elif page == pages[1]:
    st.write("## Exploratory Data Analysis")
    st.write("L'analyse exploratoire des données (EDA) est une étape cruciale dans le processus de détection de la fraude par carte de crédit.")
    st.write("Elle permet de comprendre la structure des données, d'identifier les tendances et les anomalies, et de guider les choix de modélisation.")
    
    st.dataframe(df.head())

    st.write("Dimensions Datafrme:")

    st.write(df.shape)

    if st.checkbox("Afficher les valeurs manquantes"):
        st.write(df.isna().sum())
    if st.checkbox("Afficher les statistiques descriptives"):
        st.write(df.describe())

elif page == pages[2]:
    st.markdown("""
## Data Analysis

L'analyse des données de transactions par carte de crédit met en évidence plusieurs éléments importants :

- Il existe un **fort déséquilibre** entre les classes :
  - La grande majorité des transactions sont **normales** (classe 0)
  - Les transactions **frauduleuses** (classe 1) représentent une **proportion très faible** du total

Ce déséquilibre a un impact direct sur la qualité des modèles :

- Un modèle non adapté pourrait :
  - prédire toutes les transactions comme *non frauduleuses*
  - obtenir une accuracy élevée (≈ 99 %)
  - **mais échouer totalement** à détecter les vraies fraudes

Pour traiter ce problème, il est nécessaire d’adapter la stratégie d’apprentissage :

- Techniques de gestion du déséquilibre :
  - **Oversampling**
  - **Undersampling**
  - **SMOTE** (Synthetic Minority Oversampling Technique)

Ces approches permettent d’améliorer la capacité du modèle à identifier les comportements frauduleux.
""")

    

    st.subheader("Répartition des transactions normales et frauduleuses (Interactif)")
    class_counts = df['Class'].value_counts()
    # Conversion en DataFrame pour Plotly
    plot_df = class_counts.reset_index()
    plot_df.columns = ["Classe", "Nombre"]

    # Renommage pour lisibilité
    plot_df["Classe"] = plot_df["Classe"].map({0: "Normales", 1: "Fraudes"})

    fig = px.bar(
        plot_df,
        x="Classe",
        y="Nombre",
        color="Classe",
        color_discrete_map={"Normales": "#4CAF50", "Fraudes": "#F44336"},
        title="Répartition des transactions normales et frauduleuses",
        text="Nombre"
    )

    fig.update_layout(
        xaxis_title="Classe",
        yaxis_title="Nombre de transactions",
        template="plotly_white",
    )

    st.plotly_chart(fig)

    
    
elif page == pages[3]:

    #std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df.drop(['Time','Amount'], axis=1, inplace=True)

    st.write("## Data Preprocessing")
    st.write("La préparation des données est une étape cruciale dans le processus de détection de la fraude par carte de crédit.")
    st.markdown("""   
Prochaines étapes :

- Mise à l’échelle (scaling) : Les colonnes Amount et Time sont mises à l’échelle afin qu’elles aient des valeurs comparables à celles des autres colonnes.

- Création du sous-échantillon : Le jeu de données contient 492 cas de fraude. Nous allons sélectionner aléatoirement 492 cas de non-fraude afin de créer un nouveau sous-échantillon.

- Concaténation : Les 492 cas de fraude et les 492 cas de non-fraude sont concaténés pour former un nouveau sous-échantillon équilibré.


""")
    
    card_df = df.sample(frac=1)

    # amount of fraud classes 492 rows.

    fraud_df = card_df.loc[card_df['Class'] == 1]
    non_fraud_df = card_df.loc[card_df['Class'] == 0][:492]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # shuffle dataframe rows
    new_card_df = normal_distributed_df.sample(frac=1, random_state=42)
    

    # Afficher la distribution dans la console (comme ton print original)
    #print("Distribution of the classes in the subsampled dataset")
    #print(new_card_df['Class'].value_counts() / len(new_card_df))

    # Graphique interactif
    fig = px.histogram(
    new_card_df,
    x="Class",
    color="Class",
    color_discrete_sequence=px.colors.qualitative.Set2,
    barmode="group",
    title="Equally Distributed Classes (Interactive Plot)",
)

    fig.update_layout(
    xaxis_title="Classe (0 = normale, 1 = fraude)",
    yaxis_title="Nombre d’observations",
    bargap=0.2
)

    st.plotly_chart(fig, use_container_width=True)


    st.markdown("""
                ### Matrice de correlation:

Les matrices de correlation sont essentielles pour comprendre nos donnees. Nous voulons savoir s'il existe des caracteristiques 
qui influencent fortement si une transaction specifique est frauduleuse. Cependant, il est important que nous utilisions le bon 
dataframe(sous-echantillon) afin de voir quelles caracteristiques ont une forte correlation positive ou negative par rapport aux
transaction frauduleuses.

- Les variables V17, V14, V12 et V10 présentent une corrélation négative avec la variable Class.
Cela signifie que plus leurs valeurs diminuent, plus la probabilité qu’une transaction soit frauduleuse augmente.

- Les variables V2, V4, V11 et V19 montrent une corrélation positive avec la variable Class.
Cela indique que plus leurs valeurs sont élevées, plus il est probable qu’une transaction soit frauduleuse.

- Des boxplots permettront de visualiser ces tendances en comparant la distribution de ces variables entre les transactions frauduleuses (Class = 1) et non frauduleuses (Class = 0).

- Note: Nous devons nous assurer d'utiliser le sous-echantillon dans notre matrice de correlation, sinon notre matrice de correlation sera affectee par le fort desequilibre entre nos classes. Cela se produit en raison du grand desequilibre des classes dans le Dataframe original.
                               
                """)


    # Calcul de la matrice de corrélation
    sub_sample_corr = new_card_df.corr()

    # Graphique interactif
    fig = px.imshow(
        sub_sample_corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Subsample Correlation Matrix (Interactive)",
        aspect="auto"
    )

    fig.update_layout(
        width=900,
        height=900,
    )

    st.plotly_chart(fig, use_container_width=True)   


    # Sélection des features à visualiser
    features = ["V11", "V4", "V2", "V19"]

    # Conversion au format long pour Plotly
    df_long = new_card_df.melt(
        id_vars="Class", 
        value_vars=features,
        var_name="Feature", 
        value_name="Value"
    )

    # Graphique interactif
    fig = px.box(
        df_long,
        x="Feature",
        y="Value",
        color="Class",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Boxplots interactifs des features les plus corrélées à la fraude"
    )

    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Valeurs",
        boxmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sélection des features à visualiser
    features = ["V17", "V14", "V12", "V10"]
    
    # Conversion au format long pour Plotly
    df_long = new_card_df.melt(
        id_vars="Class", 
        value_vars=features,
        var_name="Feature", 
        value_name="Value"
    )
    
    # Graphique interactif
    fig = px.box(
        df_long,
        x="Feature",
        y="Value",
        color="Class",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Valeurs",
        boxmode="group"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(""" 
                ### Detection des anomalies

Notre objectif principal dans la suite est de supprimer les "outliers extremes" des caracteristiques qui ont une forte correlation avec nos classes. Cela aura un impact sur la precision de nos modeles.
#### Methode de l'intervalle interquartile (IQR)

IQR: Nous calculons cela par la difference entre 75e percentile et la 25e percentile. Notre objectif est de creer un seuil au dela du 75e et du 25e percentile, de maniere a ce que si une instance depasse ce seuil, elle soit supprimee. Boxplots: En plus de pouvoir voir facilement les 25e et 75e percentiles, il est aussi facile de reperer les outliers extremes.

#### Compromis de la suppression des outliers:
Nous devons etre prudents quant a la distance a laquelle nous voulons definir le seuil pour supprimer les outliers. Nous determinons ce seuil en multipliant un nombre par l'intervalle interquartile(IQR). Plus ce seuil est eleve, moins d'outliers seront detectes et plus ce seuil est bas, plus il y aura d'outliers detectes.

Le compromis: Plus le seuil est bas, plus d'outliers seront supprimes, cependant, nous voulons nous concentrer davantage sur les outliers extremes plutot que sur les simples outliers. Parceque nous risquons de perdre des informations, ce qui pourrait entrainer une dimunition de la precision de nos modeles

#### Resume:

- Nous commencons par visualiser la distribution des caracteristiques que nous allons utiliser pour eliminer certains outliers. v14 est la seule caracteristique qui suit une distribution gaussienne par rapport aux caracteristiques v12 et v10.

- Apres avoir decide quel nombre nous allons utiliser pour multiplier l'IQR, nous allons determiner les seuils superieurs et inferieurs en soustrayant (q25 - seuil) pour le seuil extreme inferieur et en ajoutant (q75+ seuil) pour le seuil extreme superieur.

- Enfin nous creeons une suppression conditionnelle, indiquant que si le seuil est depasse dans les deux extremes les instances seront supprimees.

**Note**

Apres avoir applique la reduction des outliers, notre precision a ete amelioree de plus de 3%. Certains outliers peuvent fausser la precision de nos modlees mais il est important de veiller a ne pas perdre trop d'informations, si non notre modele risque de souffrir de sous-apprentissage

                """)    
    

    features = ["V14", "V12", "V10"]
    colors = ["red", "grey", "purple"]

    for feature, color in zip(features, colors):
        # Filtrer les transactions frauduleuses
        fraud_data = new_card_df.loc[new_card_df['Class'] == 1, feature].values

        # Histogramme
        hist = np.histogram(fraud_data, bins=30)
        x_hist = (hist[1][:-1] + hist[1][1:]) / 2  # centres des bins
        y_hist = hist[0]

        # Distribution normale
        mu, std = norm.fit(fraud_data)
        x_norm = np.linspace(fraud_data.min(), fraud_data.max(), 100)
        y_norm = norm.pdf(x_norm, mu, std) * len(fraud_data) * (x_hist[1]-x_hist[0])  # scaling

        # Création du graphique
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_hist, y=y_hist, name="Histogramme", marker_color=color))
        fig.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Fit Normal', line=dict(color='black')))

        fig.update_layout(
            title=f"{feature} Distribution (Fraud Transactions)",
            xaxis_title=feature,
            yaxis_title="Count",
            bargap=0.2
        )

        st.plotly_chart(fig, use_container_width=True)
   #****************************************************************************************
        ## v14 removing outliers (highest Negative correlated with Labels)
    v14_fraud = new_card_df['V14'].loc[new_card_df['Class'] == 1].values
    q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
    v14_iqr = q75 - q25
    
    v14_cut_off = v14_iqr * 1.5
    v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
    
    outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]    
    new_card_df = new_card_df.drop( new_card_df[(new_card_df['V14'] > v14_upper) | (new_card_df['V14'] < v14_lower)].index)
    
    # v12 removing outliers from fraud transactions
    
    v12_fraud = new_card_df['V12'].loc[new_card_df['Class'] == 1].values
    q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
    v12_iqr = q75 - q25
    
    v12_cut_off = v12_iqr * 1.5
    v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
    outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
    
    # removing outliers v10 feature
    v10_fraud = new_card_df['V10'].loc[new_card_df['Class'] == 1].values
    q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
    v10_iqr = q75 - q25
    
    v10_cut_off = v10_iqr * 1.5
    v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
    outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]


    #************************************************************************************


    # new_card_df is from the random undersample data (fewer instances)
    X=new_card_df.drop('Class', axis=1)
    y=new_card_df['Class']

    # T-SNE Implementation
    t0 = time.time()
    X_reduced_tsne = PCA(n_components=2, random_state = 42).fit_transform(X.values)
    t1 = time.time()
    
    # PCA implementation
    t0 = time.time()
    X_reduced_pca = PCA(n_components=2, random_state = 42).fit_transform(X.values)
    t1 = time.time()

    # TruncatedSVD
    t0 = time.time()
    X_reduced_tsvd = TruncatedSVD(n_components=2, algorithm='randomized',random_state = 42).fit_transform(X.values)
    t1 = time.time()

   

    # Créer une figure avec 1 ligne et 3 colonnes
    fig = make_subplots(
        rows=1, cols=3, 
        subplot_titles=("t-SNE", "PCA", "Truncated SVD")
    )

    # Couleurs
    colors = {0: '#3498db', 1: '#e74c3c'}

    # t-SNE
    fig.add_trace(
        go.Scatter(
            x=X_reduced_tsne[:,0], y=X_reduced_tsne[:,1],
            mode='markers',
            marker=dict(color=[colors[i] for i in y], size=5),
            name='t-SNE'
        ),
        row=1, col=1
    )

    # PCA
    fig.add_trace(
        go.Scatter(
            x=X_reduced_pca[:,0], y=X_reduced_pca[:,1],
            mode='markers',
            marker=dict(color=[colors[i] for i in y], size=5),
            name='PCA'
        ),
        row=1, col=2
    )

    # Truncated SVD
    fig.add_trace(
        go.Scatter(
            x=X_reduced_tsvd[:,0], y=X_reduced_tsvd[:,1],
            mode='markers',
            marker=dict(color=[colors[i] for i in y], size=5),
            name='Truncated SVD'
        ),
        row=1, col=3
    )

    # Mettre à jour le layout
    fig.update_layout(
        height=600, width=1200,
        showlegend=False,
        title_text="Clusters using Dimensionality Reduction"
    )

    # Ajouter une légende manuelle
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#3498db'), name='No Fraud'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#e74c3c'), name='Fraud'))

    # Afficher dans Streamlit
    st.plotly_chart(fig, width="stretch")
    print(X_reduced_tsne.dtype, X_reduced_pca.dtype, X_reduced_tsvd.dtype)
    print(np.min(X_reduced_tsne), np.max(X_reduced_tsne))
    print("T-SNE took {:.2} s", format(t1-t0))



elif page == pages[4]:
    st.markdown(""" 
                ## Modeling
                **classificateurs(Sous-echantillonnage)**:
Dans cette section nous allons entrainer 4 types de classificateurs et determiner lequel sera le plus efficacepour detecter 
les transactions frauduleuses. Avant cela, nous devons diviser nos donnees en ensembles d'entrainements et de test et separer 
les caracteristiques des etiquettes.

- Le classificateur de regression Logistique est le plus precis que les 3 autres classificateurs dans la plupart des cas. ( Nous analyserons la regression logistique plus en detail)
- GridsearchCV est utilise pour determiner les parametres qui donnent le meilleur score predictif pour les classificateurs.
- La regression logistique a le meilleur score de la caracteristique de reception operatoire (ROC) ce qui signifie qu'elle 
separe assez precisement les transactions frauduleuses et non frauduleuses.

Courbes d'apprentissage:
plus l'ecart entre le score d'entrainement et le score de validation est grand, plus il est probable que le modele fait du 
surapprentissage(variance elevee). Si le sccre est faible a la fois dans les ensembles d'entrainements et de validation croisee, 
cela indique que notre modele fait du surapprentissage (biais eleve). Le classificateur de regression logistique montre le meillleur 
score a la fois dans les ensembles d entrainements et de validation croisee

""")
    
    new_card_df = pd.read_csv("data/new_card_df.csv")
    X = new_card_df.drop('Class', axis=1)
    y = new_card_df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    classifiers = {
               "Logistic Regression": LogisticRegression(),
               "KNearest": KNeighborsClassifier(),
               "Support Vector Classifier": SVC(),
               "Decision Tree": DecisionTreeClassifier()
    }

    log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 1000]}
    #grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
    #grid_log_reg.fit(X_train, y_train)
    #log_reg = grid_log_reg.best_estimator_



    # ----- KNN -----
    knears_params = {"n_neighbors": list(range(2,5,1)),
                     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    #grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
    #grid_knears.fit(X_train, y_train)
    #knears_neighbors = grid_knears.best_estimator_

    # Sauvegarde
    #joblib.dump(knears_neighbors, "../models/KNN_best.joblib")


    # ----- SVC -----
    #svc_params = {'C': [0.5, 0.7, 0.9, 1],
    #             'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    #grid_svc = GridSearchCV(SVC(), svc_params)
    #grid_svc.fit(X_train, y_train)
    #svc = grid_svc.best_estimator_

    # Sauvegarde
    #joblib.dump(svc, "../models/SVC_best.joblib")


    # ----- Decision Tree -----
    #tree_params = {"criterion": ["gini", "entropy"],
    #             "min_samples_leaf": list(range(5,7,1))}
    #grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
    #grid_tree.fit(X_train, y_train)
    #tree_clf = grid_tree.best_estimator_

    # Sauvegarde
    #joblib.dump(tree_clf, "../models/DecisionTree_best.joblib")
    dt = joblib.load("models/DecisionTree_best.joblib")
    knn = joblib.load("models/KNN_best.joblib")
    log_reg = joblib.load("models/LogisticRegression_best.joblib")
    svc = joblib.load("models/SVC_best.joblib")

    y_pred_dt = dt.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_svc = svc.predict(X_test)

    model_chosen = st.selectbox(label = "Model", options = ["Logistic Regression", "KNearest", "Support Vector Classifier", "Decision Tree"])

    from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)



  
    def display_classification_report(y_test, y_pred):
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        st.write("###  Classification Report (format tableau)")
        st.dataframe(report_df.style.format("{:.4f}"))

    def train_model(model_name):

        # Sélection du bon vecteur de prédiction
        if model_name == "Logistic Regression":
            y_pred = y_pred_log_reg
        elif model_name == "KNearest":
            y_pred = y_pred_knn 
        elif model_name == "Support Vector Classifier":
            y_pred = y_pred_svc
        else:
            y_pred = y_pred_dt   # Decision Tree

        # --- Calcul des métriques ---
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)

        # --- Matrice de confusion ---
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"])

        # Graphique interactif Plotly
        fig_cm = ff.create_annotated_heatmap(
            z=cm_df.values,
            x=cm_df.columns.tolist(),
            y=cm_df.index.tolist(),
            showscale=True,
            colorscale="Blues"
        )

        st.plotly_chart(fig_cm)

        # --- Classification Report ---
        st.text("### Classification Report :")
        display_classification_report(y_test, y_pred)


        return accuracy, precision, recall, f1

    acc, prec, rec, f1 = train_model(model_chosen)

    st.write("###  Metrics")
    st.write(f"**Accuracy :** {acc:.4f}")
    st.write(f"**Precision :** {prec:.4f}")
    st.write(f"**Recall :** {rec:.4f}")
    st.write(f"**F1-score :** {f1:.4f}")















