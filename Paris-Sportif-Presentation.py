#importation de toutes les librairies

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier  # Import de k-NN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#lecture du csv atp ,son affichage et les différentes informations du jeux 

#atp=pd.read_csv('atp_data.csv')
#st.dataframe(atp.head())

#Création des pages

st.title("Projet de classification binaire Paris Sportif")
st.sidebar.title("Sommaire")


pages = [
    "Contexte du projet",
    "Le Jeu De Données",
    "Quelques visualisations",
    "Préparation des données",
    "Modélisation",
    "Application métier et conclusion",
    " Perspectives"
]

page = st.sidebar.radio("Aller vers la page", pages)

# Liste des auteurs dans la sidebar (avec markdown pour liste à puces)
st.sidebar.markdown("### Auteurs")
st.sidebar.markdown("""
- Nafissatou K.
- [Nom Auteur 2]
- [Nom Auteur 3]
""")


#######Page 0
if page == pages[0]:
    st.write("### Contexte du projet")
    st.markdown("""
Ce projet a été réalisé dans le cadre de notre formation en data science via l'organisme [Datascientest](https://datascientest.com).  
L'objectif est de prédire l'issue de matchs de tennis, à partir du jeu de données disponible sur [Kaggle](https://www.kaggle.com/datasets/edouardthomas/atp-matches-dataset).

Ce Streamlit présente notre démarche pour mener à bien ce projet, depuis l'exploration des données jusqu'à la création des variables explicatives.  
Les meilleurs résultats que nous avons pu obtenir y seront présentés, avec la partie Machine Learning qui vous permettra de tester vous-même les variables que nous avons créées sur différents algorithmes.
""")


    st.image("/Users/nafissatoukm/Desktop/Streamlit/cas2/tennis-paris-sportifs.jpg", use_column_width=True)


#######Page 1
if page == pages[1]:
    st.write("### 5 première lignes du Jeu de Données")
    atp = pd.read_csv('atp_data.csv')
    st.dataframe(atp.head())
    
    st.write("### Dimensions du Jeu De Données")
    st.dataframe(atp.shape)
    
    st.write("### Présence de Doublons")
    if st.checkbox("Afficher les doublons"):
        st.write("### Doublons dans le dataset ATP")
        doublons = atp[atp.duplicated()]
        if doublons.empty:
            st.write("Aucun doublon trouvé.")
        else:
            st.dataframe(doublons)

    st.write("### Le Jeu De Données")
    st.markdown("""
Ce projet s’appuie sur un jeu de données issu des compétitions ATP (Association of Tennis Professionals), 
couvrant un grand nombre de matchs professionnels de tennis. L’objectif est de prédire,
à partir de ces données historiques, la probabilité qu’un joueur A batte un joueur B — 
et ainsi tenter de faire mieux que les modèles prédictifs des bookmakers.
""")
    st.markdown("""
Voici une description détaillée des variables du dataset ATP et du deuxième csv confidence disponible sur [Kaggle](https://www.kaggle.com/datasets/edouardthomas/atp-matches-dataset).
""")

    # Données descriptives par variable
    data = {
        "N°": list(range(1, 24)),
        "Nom de la colonne": [
            "ATP", "Location", "Tournament", "Date", "Series", "Court", "Surface", "Round",
            "Best of", "Winner", "Loser", "WRank", "LRank", "Wsets", "Lsets", "Comment",
            "PSW", "PSL", "B365W", "B365L", "elo_winner", "elo_loser", "proba_elo"
        ],
        "Description": [
            "Identifiant du tournoi",
            "Lieu du match (ville ou pays)",
            "Nom du tournoi (correspondance avec ATP)",
            "Date du match (à convertir en datetime)",
            "Série du tournoi (niveau: Grand Slam, ATP 1000, etc.)",
            "Type de court (Indoor/Outdoor)",
            "Surface du match (Hard, Clay, Grass, Carpet)",
            "Tour du tournoi",
            "Format du match (meilleur de 3 ou 5 sets)",
            "Nom du joueur gagnant",
            "Nom du joueur perdant",
            "Classement mondial joueur gagnant",
            "Classement mondial joueur perdant",
            "Sets gagnés par joueur gagnant",
            "Sets gagnés par joueur perdant",
            "Issue du match (Completed, Retired, Walkover, Disqualified)",
            "Probabilité victoire joueur gagnant (modèles prédictifs)",
            "Probabilité victoire joueur perdant (modèles prédictifs)",
            "Cote bookmaker joueur gagnant (Bet365)",
            "Cote bookmaker joueur perdant (Bet365)",
            "Score Elo joueur gagnant",
            "Score Elo joueur perdant",
            "Probabilité victoire joueur gagnant (Elo)"
        ],
        "Disponible en prédiction": [
            "Oui", "Oui", "Oui", "Oui", "Oui", "Oui", "Oui", "Oui", "Oui",
            "Non", "Non", "Oui", "Oui", "Non", "Non", "Non", "Oui", "Oui",
            "Oui", "Oui", "Oui", "Oui", "Oui"
        ],
        "Type informatique": [
            "int64", "object", "object", "object", "object", "object", "object",
            "object", "int64", "object", "object", "int64", "int64", "float64",
            "float64", "object", "float64", "float64", "float64", "float64",
            "float64", "float64", "float64"
        ],
        "Taux de NA (%)": [
            "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
            "41.8", "41.8", "0", "26.8", "26.8", "12.7", "12.6", "0", "0", "0"
        ],
        "Gestion des NA": [
            "—", "—", "—", "—", "—", "—", "—", "—", "—",
            "—", "—", "—", "—", "Suppression ou moyenne", "Suppression ou moyenne", "—",
            "Remplacement par médiane", "Remplacement par médiane", "Remplacement par moyenne",
            "Remplacement par moyenne", "—", "—", "—"
        ],
        "Commentaires / Distribution": [
            "Quantitative, pas de NA",
            "Catégorielle (>10 catégories), pas de NA",
            "Catégorielle (>10 catégories), pas de NA",
            "Date à convertir au format datetime",
            "Catégorielle, 8 catégories: International, Grand Slam, etc.",
            "Catégorielle, 2 catégories: Indoor, Outdoor",
            "Catégorielle, 4 catégories: Hard, Clay, Grass, Carpet",
            "Catégorielle, 8 catégories (tours du tournoi)",
            "Quantitative, pas de NA",
            "Catégorielle, non disponible avant match",
            "Catégorielle, non disponible avant match",
            "Quantitative, pas de NA",
            "Quantitative, pas de NA",
            "Quantitative, NA ~42%, convertir float -> int",
            "Quantitative, NA ~42%, convertir float -> int",
            "Catégorielle, 4 modalités: Completed, Retired, Walkover, Disqualified",
            "Quantitative, NA > 20%, remplacer par médiane",
            "Quantitative, NA > 20%, remplacer par médiane",
            "Quantitative, NA ~13%, remplacer par moyenne",
            "Quantitative, NA ~13%, remplacer par moyenne",
            "Quantitative, pas de NA",
            "Quantitative, pas de NA",
            "Quantitative, pas de NA"
        ]
    }

    df = pd.DataFrame(data)

    st.subheader("Description détaillée des variables ATP")
    st.table(df)

    st.markdown("""
---
### Informations supplémentaires du deuxième dataset confidence

| N° | Col           | Description                               | Disponibilité de la variable a priori | Type informatique | Taux de NA | Gestion des NA                     | Distribution des valeurs | Remarques sur la colonne                                         |
|----|---------------|-----------------------------------------|-------------------------------------|-------------------|------------|-----------------------------------|-------------------------|-----------------------------------------------------------------|
| 1  | match         | Identifiant ou description du match     | oui                                 | int64             | 0.00000    | —                                 | Quantitative            |                                                                 |
| 2  | PSW           | Probabilité de victoire du joueur gagnant | oui                                 | float64           | 0.416139   | <5% supprimer ou remplacer par la moyenne | Quantitative            | Peut être utilisée pour fusion mais induit des erreurs au niveau des dates |
| 3  | win0          | 1 = victoire, 0 = défaite                | oui                                 | int64             | 0.00000    | —                                 | Quantitative            | Potentielle variable cible                                       |
| 4  | confidence0   | Confiance du modèle dans la prédiction  | oui                                 | float64           | 0.00000    | —                                 | Quantitative            |                                                                 |
| 5  | date          | Date à laquelle le match a eu lieu       | oui                                 | object            | 0.00000    | —                                 | —                       | À convertir en datetime                                          |

---

""")

    st.markdown("""
La fusion avec des deux datasets a été envisagée puis abondonnée car complexe et génére des NA importants.
""")

#######Page 2
# Charger les données une seule fois et les cacher pour optimiser
@st.cache_data
def load_data():
    return pd.read_csv('atp_data.csv')

atp = load_data()

if page == pages[2]:
    st.write("### Quelques visualisations")
    fig = px.scatter(
        atp,
        x='B365W',
        y='PSW',
        title='Comparaison entre B365W et PSW en fonction des athlètes gagnants',
        labels={'B365W': 'Cote Bookmaker B365W', 'PSW': 'Probabilité Modèle PSW'},
        color='Winner'
    )
    st.plotly_chart(fig, use_container_width=True)

#######Page 3
# Fonction de création de la cible
def create_target_with_random_inversion(data, seed=None):
    expected_cols = ['Winner', 'Loser', 'WRank', 'LRank', 'Wsets', 'Lsets', 'elo_winner', 'elo_loser']
    missing_cols = [col for col in expected_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")
    
    if seed is not None:
        np.random.seed(seed)

    targets = []
    inversions = np.random.rand(len(data)) < 0.5

    for i in range(len(data)):
        row = data.iloc[i]
        if inversions[i]:
            data.at[i, 'PlayerA'] = row['Loser']
            data.at[i, 'PlayerB'] = row['Winner']
            data.at[i, 'RankA'] = row['LRank']
            data.at[i, 'RankB'] = row['WRank']
            data.at[i, 'SetsA'] = row['Lsets']
            data.at[i, 'SetsB'] = row['Wsets']
            data.at[i, 'EloA'] = row['elo_loser']
            data.at[i, 'EloB'] = row['elo_winner']
            targets.append(0)
        else:
            data.at[i, 'PlayerA'] = row['Winner']
            data.at[i, 'PlayerB'] = row['Loser']
            data.at[i, 'RankA'] = row['WRank']
            data.at[i, 'RankB'] = row['LRank']
            data.at[i, 'SetsA'] = row['Wsets']
            data.at[i, 'SetsB'] = row['Lsets']
            data.at[i, 'EloA'] = row['elo_winner']
            data.at[i, 'EloB'] = row['elo_loser']
            targets.append(1)
    data['Target'] = targets

# Page 3 - Préparation des données
if page == pages[3]:
    st.title("🧼 Préparation des données")

    st.markdown("""
## Objectif :
Nettoyage et transformation du jeu de données pour permettre une modélisation efficace.
""")

    # Conversion de la date
    atp['Date'] = pd.to_datetime(atp['Date'])
    atp['Year'] = atp['Date'].dt.year
    st.markdown("### 🔄 Conversion des types")
    st.dataframe(atp[['Date', 'Year']].head())

    st.markdown("### 🔄 Pourcentages de NA par colonne")
    # Calcul des pourcentages de NA par colonne
    manquants = atp.isna().mean() * 100
    manquants = manquants[manquants > 0]  # filtrer seulement colonnes avec NA > 0

    st.dataframe(manquants.sort_values(ascending=False))

    # Graphique barres des NA
    fig, ax = plt.subplots(figsize=(10,6))
    manquants.sort_values(ascending=False).plot(kind='bar', color='orange', ax=ax)
    ax.set_ylabel("Pourcentage de valeurs manquantes (%)")
    ax.set_title("Taux de valeurs manquantes par colonne")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig)


    # Création de la cible et inversion des joueurs
    st.markdown("### 🎯 Création de la variable cible `Target` avec inversion aléatoire")
    create_target_with_random_inversion(atp, seed=42)
    st.dataframe(atp[['PlayerA', 'PlayerB', 'Target']].sample(5))

    # Distribution de Target
    st.write("**Distribution de la variable `Target` :**")
    st.bar_chart(atp['Target'].value_counts())

    # Feature engineering
    st.markdown("### 🛠️ Feature Engineering")
    atp['Rank_Diff'] = atp['RankA'] - atp['RankB']
    atp['Elo_Diff'] = atp['EloA'] - atp['EloB']
    st.dataframe(atp[['RankA', 'RankB', 'Rank_Diff', 'EloA', 'EloB', 'Elo_Diff']].head())

    # Visualisation des distributions
    st.markdown("### 📊 Visualisation des nouvelles variables")

    fig1, ax1 = plt.subplots()
    sns.histplot(atp['Rank_Diff'], bins=50, kde=True, ax=ax1)
    ax1.set_title("Distribution de Rank_Diff")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.histplot(atp['Elo_Diff'], bins=50, kde=True, color='orange', ax=ax2)
    ax2.set_title("Distribution de Elo_Diff")
    st.pyplot(fig2)

#######Page 4
if page == pages[4]:
    st.title("Modélisation")

    st.markdown("""
    Voici les performances des modèles sur le jeu de test, avec accuracy et matrices de confusion.
    """)

    # Chargement des données test (ajuste les chemins si besoin)
    X_test_final = pd.read_csv("X_test_final.csv")
    y_test = pd.read_csv("y_test.csv").values.ravel()

    # Chargement des modèles
    rf_model = joblib.load('rf_model.joblib')
    lr_model = joblib.load('lr_model.joblib')
    dtc_model = joblib.load('dtc_model.joblib')
    knn_model = joblib.load('knn_model.joblib')

    # Fonction pour afficher matrice de confusion
    def plot_confusion_matrix(y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel('Prédictions')
        ax.set_ylabel('Vraies étiquettes')
        ax.set_title(f'Matrice de confusion - {model_name}')
        st.pyplot(fig)

    # Menu déroulant pour choisir un modèle
    model_choice = st.selectbox(
        "Sélectionnez un modèle à afficher :",
        ("Random Forest", "Régression Logistique", "Decision Tree", "K-Nearest Neighbors")
    )

    # Dictionnaire des modèles et leurs prédictions
    models = {
        "Random Forest": rf_model.predict(X_test_final),
        "Régression Logistique": lr_model.predict(X_test_final),
        "Decision Tree": dtc_model.predict(X_test_final),
        "K-Nearest Neighbors": knn_model.predict(X_test_final)
    }

    # Affichage des résultats du modèle sélectionné
    preds = models[model_choice]
    acc = accuracy_score(y_test, preds)
    st.write(f"### {model_choice}")
    st.write(f"Accuracy : {acc:.3f}")
    plot_confusion_matrix(y_test, preds, model_choice)

        
#######Page 5
if page == pages[5]:
    st.title("📈 Application métier et conclusion")

    st.markdown("""
## 🎯 Application métier

Dans le cadre de ce projet, nous avons conçu une stratégie de **paris sportifs** fondée sur l’identification des *outsiders* — c’est-à-dire les équipes considérées comme moins probables de gagner par les bookmakers — et sur l'agrégation des prédictions de plusieurs modèles de machine learning.

L’application métier est double :
- Optimiser la prise de décision en matière de paris en se concentrant sur les **matchs à forte valeur attendue**.
- Exploiter l’intelligence collective de modèles (**vote majoritaire**) pour maximiser la fiabilité des prédictions, tout en prenant en compte les **cotes proposées** (via un filtre sur les cotes minimales).

### 🧪 Deux stratégies simulées :
1. **Stratégie prudente** : parier uniquement lorsque **tous les modèles (4/4)** sont d'accord.
2. **Stratégie opportuniste** : parier dès que **3 modèles sur 4** sont d'accord et que la **cote ≥ 2.5**, favorisant les gains plus élevés.

### 📅 Données utilisées :
- **Jeu d'entraînement** : saisons **2000 à 2011**
- **Jeu de test** : saisons **2012 à 2018**

Cette séparation temporelle respecte une logique de **prévision réelle** (pas de data leakage), simulant un vrai usage métier.
""")

    # Données sous forme de dictionnaire
    data = {
        "Jeu de données": ["Entraînement", "Entraînement", "Test", "Test"],
        "Confiance": ["100 % (4/4)", "75 % (3/4)", "100 % (4/4)", "75 % (3/4)"],
        "Cote min.": ["0", "≥ 2.5", "0", "≥ 2.5"],
        "Nombre de paris": [4573, 886, 2557, 745],
        "Mise (€)": [4573.00, 886.00, 2557.00, 745.00],
        "Gains (€)": [90.62, 2889.86, 1359.68, 3177.64],
        "ROI (%)": [1.98, 326.17, 53.17, 426.53],
        "% Outsiders corrects": ["29.79 %", "29.79 %", "26.95 %", "26.95 %"]
    }

    df_stats = pd.DataFrame(data)

    # Affichage Streamlit
    st.markdown("## 📊 Résultats des performances")
    st.dataframe(df_stats)

    st.markdown("""
## ✅ Conclusion

Ce projet démontre la possibilité de générer des retours sur investissement positifs en exploitant l’**intelligence collective de modèles prédictifs** et des filtres stratégiques sur les cotes.

La méthode respecte la rigueur d’un **scénario métier réaliste**, sans fuite d’informations, et montre un fort potentiel d'application dans l’aide à la décision pour les parieurs.

""")
if page == pages[6]:
    st.title("🧭 Critique et perspectives")

    st.markdown("""
## ❗️6.6 Limites du projet

Même si les résultats sont prometteurs, plusieurs **limites** doivent être soulignées :

- **Sélection manuelle des features** : le projet repose sur des variables simples (classements, ELO, etc.). D'autres dimensions (surface, fatigue, historiques H2H, météo, etc.) n'ont pas été intégrées.
- **Pas de tuning hyperparamètres** approfondi : les modèles ont été entraînés avec des paramètres par défaut ou une grille limitée. Une optimisation plus poussée aurait pu améliorer les scores.
- **Absence de validation croisée temporelle** : le découpage temporel est réaliste, mais ne couvre qu’un seul split train/test. Une validation croisée glissante aurait permis de mieux mesurer la robustesse.
- **Modèles statiques** : les modèles ne sont pas recalibrés saison après saison, ce qui réduit leur capacité à s’adapter à des évolutions récentes dans le sport.

""")

    st.markdown("""
## 🌱 6.7 Perspectives d’amélioration

Avec plus de temps, plusieurs axes d'amélioration sont envisageables :

- **Feature engineering avancé** :
  - Ajout de variables contextuelles : surface, météo, blessure, historique de confrontation, fatigue (nombre de sets/matchs récents).
  - Intégration de cotes plus précises (ouverture, live) pour mieux quantifier la "valeur".
  - Calcul de tendances ou dynamiques récentes (forme glissante sur les 5 derniers matchs, par exemple).

- **Modèles plus puissants** :
  - Test de modèles avancés comme XGBoost, LightGBM, CatBoost ou même des réseaux de neurones tabulaires.
  - Exploration de modèles probabilistes pour estimer des distributions plutôt que des classes binaires.

- **Validation plus rigoureuse** :
  - Mise en place d’une **validation croisée temporelle glissante** pour simuler plusieurs années de paris en continu.
  - Backtesting plus complet avec gestion dynamique du capital et simulations de mises variables.

- **Application métier automatisée** :
  - Développement d’une interface complète pour le suivi en temps réel des matchs à parier.
  - Déploiement sur API ou application mobile pour alerter les utilisateurs en temps réel.
  - Intégration avec des plateformes de paris pour automatiser les décisions.

""")

    st.markdown("""
## 🔄 En résumé

Ce projet pose les **fondations solides** d'une stratégie de paris basée sur la donnée, mais **de nombreuses améliorations restent possibles** pour passer à l’échelle.  
Avec plus de temps, on pourrait aller vers une **solution automatisée, dynamique et intelligente**, capable de détecter des paris à forte valeur en continu.
""")
