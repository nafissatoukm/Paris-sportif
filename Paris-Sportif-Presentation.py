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
- Nafissatou KODA
- Samir EMBAREK 
- Zacharie PALOU 
- Jordan TOUSSAINT

""")


#######Page 0
if page == pages[0]:
    st.write("### Contexte du projet")
    st.markdown("""
Ce projet a été réalisé dans le cadre de notre formation en data science via l'organisme [Datascientest](https://datascientest.com).  
L'objectif est de prédire l'issue de matchs de tennis, à partir du jeu de données disponible sur [Kaggle](https://www.kaggle.com/datasets/edouardthomas/atp-matches-dataset).

Ce Streamlit présente notre démarche pour mener à bien ce projet, depuis l'exploration des données jusqu'à la création des variables explicatives.  
Les meilleurs résultats que nous avons pu obtenir y seront présentés, avec la partie Machine Learning.
""")


    st.image("images/tennis-paris-sportifs.jpg", use_container_width=True)



#######Page 1

# ---------- Descriptions des colonnes ----------
descriptions = {
    "atp": {
        "ATP":"l'identifiant du tournoi", 
        "Location": "Le lieu où le match a eu lieu (ville ou pays).",
        "Tournament" : "Le nom du tournoi.",
        "Date": "Date du match.",
        "Série": "Le niveau du tournoi (Grand Slam, ATP 1000, ATP 500, ATP 250).",
        "Court": "Le type de court.",
        "Surface": "La surface sur laquelle le match a eu lieu.",
        "Round": "Le tour du tournoi.",
        "Best of": "Le format du match (ex: 3 sets, 5 sets).",
        "Winner": "Nom du joueur gagnant.",
        "Loser": "Nom du joueur perdant.",
        "WRank": "Classement du joueur gagnant.",
        "LRank": "Classement du joueur perdant.",
        "Wsets": "Sets gagnés par le vainqueur.",
        "Lsets": "Sets perdus par le vainqueur.",
        "Comment": "Indique l'issue du match (ex: abandon).",
        "PSW": "Probabilité de victoire du joueur gagnant.",
        "PSL": "Probabilité de victoire du joueur perdant.",
        "B365W": "Cote Bookmaker (gagnant).",
        "B365L": "Cote Bookmaker (perdant).",
        "elo_winner": "Score Elo du gagnant.",
        "elo_loser": "Score Elo du perdant.",
        "proba_elo": "Proba victoire selon Elo."
    },
    "atp_confidence": {
        "match": "Identifiant du match.",
        "PSW": "Probabilité de victoire du joueur gagnant.",
        "win0": "1 = victoire, 0 = défaite",
        "confidence0": "Confiance du modèle dans la prédiction.",
        "date": "Date du match."
    }
}

# ---------- Page 1 : Exploration des données ----------
if page == pages[1]:
    # ---------- Chargement des datasets ----------
    try:
        atp = pd.read_csv('atp_data.csv')
        st.success(" Dataset 'atp_data.csv' chargé avec succès.")
    except FileNotFoundError:
        st.error(" Fichier 'atp_data.csv' introuvable.")
        atp = pd.DataFrame()

    try:
        atp_confidence = pd.read_csv('confidence_data.csv')
        st.success(" Dataset 'confidence_data.csv' chargé avec succès.")
    except FileNotFoundError:
        st.error(" Fichier 'confidence_data.csv' introuvable.")
        atp_confidence = pd.DataFrame()

    # ---------- Dictionnaire des datasets ----------
    datasets = {
        "atp": atp,
        "atp_confidence": atp_confidence
    }

    st.header(" Exploration des données")

    # Sélection du dataset
    choix = st.selectbox("Choisissez un dataset à afficher :", ["Liste des Datasets"] + list(datasets.keys()))

    if choix != "Liste des Datasets":
        df = datasets[choix]

        if df.empty:
            st.error("⚠️ Le dataset sélectionné est vide ou n’a pas pu être chargé.")
        else:
            # Conversion automatique de la colonne 'Date' ou 'date' en datetime
            for date_col in ["Date", "date"]:
                if date_col in df.columns:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        st.success(f" Colonne '{date_col}' convertie en datetime.")
                    except Exception as e:
                        st.error(f"⚠️ Impossible de convertir '{date_col}' en datetime : {e}")

            st.subheader(f" Aperçu du dataset : {choix}")
            st.dataframe(df.head())
            st.success(f" Dimensions : {df.shape[0]} lignes - {df.shape[1]} colonnes")

            # Valeurs manquantes
            if st.checkbox(" Afficher les valeurs manquantes"):
                na_counts = df.isna().sum()
                na_counts = na_counts[na_counts > 0]
                if not na_counts.empty:
                    st.dataframe(na_counts)
                    st.warning(f" Colonnes avec valeurs manquantes : {na_counts.shape[0]}")
                else:
                    st.info(" Aucune valeur manquante détectée.")

            # Types de données
            if st.checkbox("Afficher les types de données (dtypes)"):
                st.dataframe(df.dtypes)

            # Statistiques descriptives
            if st.checkbox("Afficher les statistiques descriptives"):
                stats = df.describe()
                st.dataframe(stats)

                # Vérification des WRank / LRank suspects
                for col in ["WRank", "LRank"]:
                    if col in stats.columns and stats.loc["min", col] == 0:
                        st.warning(f" '{col}' contient un minimum de 0 → joueurs non classés ?")

            # Description colonne par colonne
            if choix in descriptions:
                colonnes = list(descriptions[choix].keys())
                colonne_choisie = st.selectbox(" Choisissez une colonne pour voir sa description :", colonnes)
                st.markdown(f"** {colonne_choisie} :** {descriptions[choix][colonne_choisie]}")
            else:
                st.info(" Aucune description disponible pour ce dataset.")
    else:
        st.info(" Aucun dataset sélectionné.")

#######Page 2
# Charger les données une seule fois et les cacher pour optimiser
@st.cache_data
def load_data():
    return pd.read_csv('atp_data.csv')

atp = load_data()


if page == pages[2]:
    st.write("### Quelques visualisations")
    st.write("Nous affichons ici quelques visualisations graphiques afin de présenter le contexte et les premières analyses des variables qui nous permettrons d'élaborer les premières actions en vue de la modélisation")
    choix = [
        "Répartition du nombre de matchs par an et par type de tournois",
        "Top 30",
        "Nuage de points des cotes des bookmakers",
        "Heatmap de corrélation",
        "Courbe sigmoïde d'écart entre les scores Elo",
        "Graphique de dispersion entre WRank et Round",
        "Box plot classement des vainqueurs par surface et tour",
    ]
    option = st.selectbox('Choix de la visualisation', choix)
    st.write('Visualisation en cours:', option)
    # Extraction de l'année à partir de la colonne Date si ce n'est pas déjà fait
    if 'Year' not in atp.columns:
        atp['Date'] = pd.to_datetime(atp['Date'])
        atp['Year'] = atp['Date'].dt.year
    if option == "Répartition du nombre de matchs par an et par type de tournois":
        Series={"International":"ATP250", "International Gold":"ATP500", "Masters Cup":"Masters"}
        atp["Series"]=atp["Series"].replace(Series)
        fig = plt.figure(figsize=(15,8))
        sns.countplot(data=atp, x="Year", hue="Series")
        plt.xticks(rotation=90)
        plt.legend(loc="best")
        plt.title("Répartition du nombre de matchs par an et par type de tournois")
        st.pyplot(fig)
    if option == "Top 30":
        top_players = atp["Winner"].value_counts().head(30)
        bad_players = atp["Loser"].value_counts().head(30)
        fig = plt.figure(figsize=(15,10))
        plt.subplot(1,2,1)
        sns.barplot(x=top_players.index, y=top_players.values, color="blue")
        plt.xticks(rotation=75)
        plt.title("Top 30 des joueurs vainqueurs de matchs")
        plt.subplot(1,2,2)
        sns.barplot(x=bad_players.index, y=bad_players.values, color="red")
        plt.xticks(rotation=75)
        plt.title("Top 30 des joueurs perdants de matchs")
        st.pyplot(fig)
    if option == "Nuage de points des cotes des bookmakers":
        fig = px.scatter(
            atp,
            x='B365W',
            y='PSW',
            title='Comparaison entre B365W et PSW en fonction des athlètes gagnants',
            labels={'B365W': 'Cote Bookmaker B365W', 'PSW': 'Probabilité Modèle PSW'},
            color='Winner'
        )
        st.plotly_chart(fig, use_container_width=True)
    if option == "Heatmap de corrélation":
        fig = plt.figure(figsize=(15,8))
        corr = atp[["Best of", "WRank", "LRank", "Wsets", "Lsets", 'PSW', 'PSL', 'B365W', 'B365L', 'elo_winner', 'elo_loser', 'proba_elo']].corr()
        sns.heatmap(data=corr, annot=True, cmap="rocket")
        st.pyplot(fig)
    if option == "Courbe sigmoïde d'écart entre les scores Elo":
        atp["elo_diff"] = atp["elo_winner"] - atp["elo_loser"]
        fig = plt.figure(figsize=(15, 10))
        sns.scatterplot(x=atp['elo_diff'], y=atp['proba_elo'], color='blue', alpha=0.5)
        plt.xlabel('Différence Elo (elo_winner - elo_loser)')
        plt.ylabel('Proba Elo')
        plt.title('Relation entre la différence Elo et la probabilité Elo')
        st.pyplot(fig)
    if option == "Graphique de dispersion entre WRank et Round":
        round_order = {
            'Round Robin': 0, '1st Round': 1, '2nd Round': 2, '3rd Round': 3, '4th Round': 4,
            'Quarterfinals': 5, 'Semifinals': 6, 'The Final': 7
        }
        atp['Round_Encoded'] = atp['Round'].map(round_order)
        fig = plt.figure(figsize=(15,8))
        sns.scatterplot(data=atp, x='WRank', y="Round_Encoded", color='blue', label="Matchs", s=100, alpha=0.6)
        plt.title("Relation entre le classement du gagnant (WRank) et le round du tournoi")
        plt.xlabel("WRank (Classement du gagnant)")
        plt.ylabel("Round (Tour du tournoi)")
        plt.ylim(0.5,6.5)
        st.pyplot(fig)
    if option == "Box plot classement des vainqueurs par surface et tour":
        atp_tournoi_majeur = atp.loc[atp['Series'].isin(['Grand Slam', 'Masters', 'Masters 1000'])]
        atp_tournoi_mineur = atp.loc[atp['Series'].isin(['ATP250', 'ATP500'])]
        fig = plt.figure(figsize=(25,15))
        ordre_tours = ['1st Round', '2nd Round', '3rd Round', '4th Round','Quarterfinals', 'Semifinals', 'The Final', 'Round Robin']
        ordre_tours_mineurs = ['1st Round', '2nd Round', '3rd Round', 'Quarterfinals', 'Semifinals', 'The Final', 'Round Robin']
        plt.subplot(1,2,1)
        sns.boxplot(data=atp_tournoi_majeur, x='Round', y='WRank', hue='Surface', order=ordre_tours)
        plt.title("Tournois Majeurs : Dispersion des Classements des Gagnants par Surface et Tour")
        plt.xlabel("Tour du Tournoi")
        plt.ylabel("Classement du Gagnant")
        plt.legend(title="Surface")
        plt.subplot(1,2,2)
        sns.boxplot(data=atp_tournoi_mineur, x='Round', y='WRank', hue='Surface', order=ordre_tours_mineurs)
        plt.title("Tournois mineurs : Dispersion des Classements des Gagnants par Surface et Tour")
        plt.xlabel("Tour du Tournoi")
        plt.ylabel("Classement du Gagnant")
        plt.legend(title="Surface")
        st.pyplot(fig)


#######Page 3
# Fonction de création de la cible
if page == pages[3]:

   st.write("Dans cette partie nous présentons les étapes de nettoyage et pré-processing des données en vue de réaliser les itérations de modélisation")
   if 'Year' not in atp.columns :
    atp['Date'] = pd.to_datetime(atp['Date'])
    atp['Year'] = atp['Date'].dt.year
    
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
   
    # Extraction de l'année à partir de la colonne Date si ce n'est pas déjà fait
   if 'Year' not in atp.columns :
        atp['Date'] = pd.to_datetime(atp['Date'])
        atp['Year'] = atp['Date'].dt.year

   st.title("Préparation des données")

   st.markdown("""
## Objectif :
Nettoyage et transformation du jeu de données pour permettre une modélisation efficace.
""")

   st.markdown("###  Conversion des types")
   st.write("Conversion de la date en format datetime puis création de la variable année qui sera utilisé par les modèles de Machine Learning")
   st.dataframe(atp[['Date', 'Year']].head().style.format({'Year': '{:.0f}',
        'Date': lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else ''}))

   
   st.markdown("###  Gestion des NaN")
   
   if st.checkbox("Afficher les NaNs") :
      st.dataframe(atp.isna().sum().to_frame(name="Nb de NaNs"))
   
   st.write("Pourcentages de NaN par colonne")
    # Calcul des pourcentages de NA par colonne
   manquants = atp.isna().mean() * 100
   manquants = manquants[manquants > 0]  # filtrer seulement colonnes avec NA > 0
   # Convertir en DataFrame avec nom de colonne
   manquants_df = manquants.sort_values(ascending=False).to_frame(name="Taux de valeurs manquantes (%)")
   st.dataframe(manquants_df)

   # Graphique barres des NA
   fig, ax = plt.subplots(figsize=(10,6))
   manquants.sort_values(ascending=False).plot(kind='bar', color='orange', ax=ax)
   ax.set_ylabel("Pourcentage de valeurs manquantes (%)")
   ax.set_title("Taux de valeurs manquantes par colonne")
   plt.xticks(rotation=45, ha='right')
   plt.tight_layout()
   st.pyplot(fig)

   st.markdown("###  Nettoyage du JDD")

   ###---- recherches de doublons de lignes---
   if st.checkbox("Afficher les doublons") :
      doublons_atp=atp.duplicated().sum()
      st.write("Nb de doublons dans le DF:", doublons_atp)
   ###---- recherches du nombre de modalités prises par chaque variable --
   if st.checkbox("Afficher le nombre de modalités prises par chaque variable") :
      st.dataframe(atp.nunique().to_frame("Nb d'occurences"))

   # Création de la cible et inversion des joueurs
   st.markdown("###  Création de la variable cible `Target` avec inversion aléatoire")
   st.write("La fonction a pour but ici de modifier directement le DataFrame en ajoutant une colonne cible binaire et en inversant aléatoirement les informations des joueurs A et B. Le joueur A est assigné 1 s'il est le gagnant, sinon 0.")

   create_target_with_random_inversion(atp, seed=42)
   st.dataframe(atp[['PlayerA', 'PlayerB', 'Target']].sample(5))

   # Distribution de Target
   st.write("**Distribution de la variable `Target` :**")
   st.bar_chart(atp['Target'].value_counts())

   # Feature engineering
   st.markdown("###  Feature Engineering")
   st.write("Création des différentiels de classement et score elo entre les joueurs afin d'enrichir notre modèle de machine learning et éviter les biais d'apprentissage. Cela permettra au modèle de relativiser la valeur absolue du classement / score elo du joueur par rapport à son advsersaire")
   atp['Rank_Diff'] = atp['RankA'] - atp['RankB']
   atp['Elo_Diff'] = atp['EloA'] - atp['EloB']
   st.dataframe(atp[['RankA', 'RankB', 'Rank_Diff', 'EloA', 'EloB', 'Elo_Diff']].head())

   # Visualisation des distributions
   st.markdown("###  Visualisation des nouvelles variables")

   fig1, ax1 = plt.subplots()
   sns.histplot(atp['Rank_Diff'], bins=50, kde=True, ax=ax1)
   ax1.set_title("Distribution de Rank_Diff")
   st.pyplot(fig1)

   fig2, ax2 = plt.subplots()
   sns.histplot(atp['Elo_Diff'], bins=50, kde=True, color='orange', ax=ax2)
   ax2.set_title("Distribution de Elo_Diff")
   st.pyplot(fig2)

   # Suppression des classements = 0
   st.markdown("###  Suppression des valeurs nulles des classements")
   st.code("""tennis=tennis.loc[(tennis["RankA"]!=0)&(tennis["RankB"]!=0)]""", language="python")

   # Suppression des variables inutilisables
   st.markdown("### ✂️ Suppression des variables inutilisables")
   st.code("""atp = atp.drop(['RankA','RankB',
               'Date','PlayerA','PlayerB','SetsA','SetsB','EloA','EloB','PSW', 'PSL','ATP','WRank','LRank',
               'proba_elo', 'elo_winner', 'elo_loser','Winner','Loser','Wsets','Lsets',], axis=1""", language="python")

#######Page 4
if page == pages[4]:
    st.title("Modélisation")

    st.markdown("""
    Voici les performances des modèles sur le jeu de test, avec accuracy et matrices de confusion.
    """)

    # Chargement des données test 
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
    st.title("Application métier et conclusion")

    st.markdown("""
##  Application métier

Dans le cadre de ce projet, nous avons conçu une stratégie de **paris sportifs** fondée sur l’identification des *outsiders* — c’est-à-dire les joueurs considérées comme moins probables de gagner par les bookmakers — et sur l'agrégation des prédictions de plusieurs modèles de machine learning.

L’application métier est double :
- Optimiser la prise de décision en matière de paris en se concentrant sur les **matchs à forte valeur attendue**.
- Exploiter l’intelligence collective de modèles (**vote majoritaire**) pour maximiser la fiabilité des prédictions, tout en prenant en compte les **cotes proposées** (via un filtre sur les cotes minimales).

### Deux stratégies simulées :
1. **Stratégie prudente** : parier uniquement lorsque **tous les modèles (4/4)** sont d'accord.
2. **Stratégie opportuniste** : parier dès que **3 modèles sur 4** sont d'accord et que la **cote ≥ 2.5**, favorisant les gains plus élevés.

### Données utilisées :
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
    st.markdown("##  Résultats des performances")
    st.dataframe(df_stats)

    st.markdown("""
## Conclusion

Ce projet démontre la possibilité de générer des retours sur investissement positifs en exploitant l’**intelligence collective de modèles prédictifs** et des filtres stratégiques sur les cotes.

La méthode respecte la rigueur d’un **scénario métier réaliste**, sans fuite d’informations, et montre un fort potentiel d'application dans l’aide à la décision pour les parieurs.

""")
if page == pages[6]:
    st.title("Critique et perspectives")

    st.markdown("""
## ❗️6.6 Limites du projet

Même si les résultats sont prometteurs, plusieurs **limites** doivent être soulignées :

- **Sélection manuelle des features** : le projet repose sur des variables simples (classements, ELO, etc.). D'autres dimensions (surface, fatigue, historiques H2H, météo, etc.) n'ont pas été intégrées.
- **Modèles statiques** : les modèles ne sont pas recalibrés saison après saison, ce qui réduit leur capacité à s’adapter à des évolutions récentes dans le sport.

""")

    st.markdown("""
##  6.7 Perspectives d’amélioration

Avec plus de temps, plusieurs axes d'amélioration sont envisageables :

- **Feature engineering avancé** :
  - Ajout de variables contextuelles : surface, météo, blessure, historique de confrontation, fatigue (nombre de sets/matchs récents).
  - Calcul de tendances ou dynamiques récentes (forme glissante sur les 5 derniers matchs, par exemple).

""")

    st.markdown("""
##  En résumé

Ce projet pose les **fondations solides** d'une stratégie de paris basée sur la donnée, mais **de nombreuses améliorations restent possibles** pour passer à l’échelle.  
Avec plus de temps et de connaissances, on pourrait aller vers une **solution automatisée, dynamique et intelligente**, capable de détecter des paris à forte valeur en continu.
""")
