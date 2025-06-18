# %%
#importation de pandas sous son alias
import pandas as pd

#lecture du csv atp ,son affichage et les différentes informations du jeux 
atp=pd.read_csv('atp_data.csv')
display(atp.head())
display(atp.info())
atp.shape

# %%
###---- recherches de NAN ou valeurs manquantes---
## première méthode 
# recherche du % de NAN ou valeurs manquantes dans atp
manquants_atp=atp.isna().mean()*100
display(manquants_atp)


# %%
###---- recherches de doublons de lignes---
doublons_atp=atp.duplicated().sum()
print(doublons_atp)
###---- recherches du nombre de modalités prises par chaque variable --
atp.nunique()

# %%
###---- convertion des types---
atp['Date'] = pd.to_datetime(atp['Date'])
#atp['Wsets'] = atp['Wsets'].astype(int)
#atp['Lsets'] = atp['Lsets'].astype(int)
#vérification
display(atp.dtypes)
display(atp.head())

# %%
import numpy as np
def create_target_with_random_inversion(data, seed=None):
    """
    Modifie directement le DataFrame 'data' en ajoutant une colonne cible binaire
    et en inversant aléatoirement les informations des joueurs A et B.
    Le joueur A est assigné 1 s'il est le gagnant, 0 sinon.

    :param data: DataFrame contenant les infos des matchs, avec les colonnes 'Winner', 'Loser', etc.
    :param seed: (Optionnel) Entier pour fixer la graine aléatoire (reproductibilité).
    """
    
    expected_cols = ['Winner', 'Loser', 'WRank', 'LRank', 'Wsets', 'Lsets', 'elo_winner', 'elo_loser']
    missing_cols = [col for col in expected_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing_cols}")

    # Optionnel : fixer la graine pour garantir la reproductibilité
    if seed is not None:
        np.random.seed(seed)

    # Crée la liste pour la colonne "Target"
    targets = []
    
    # Inversion aléatoire des rôles des joueurs
    inversions = np.random.rand(len(data)) < 0.5

    # On parcourt les indices du DataFrame et on applique l'inversion si nécessaire
    for i in range(len(data)):
        row = data.iloc[i]
        if inversions[i]:  # Si inversion, on échange les rôles
            data.at[i, 'PlayerA'] = row['Loser']
            data.at[i, 'PlayerB'] = row['Winner']
            data.at[i, 'RankA'] = row['LRank']
            data.at[i, 'RankB'] = row['WRank']
            data.at[i, 'SetsA'] = row['Lsets']
            data.at[i, 'SetsB'] = row['Wsets']
            data.at[i, 'EloA'] = row['elo_loser']
            data.at[i, 'EloB'] = row['elo_winner']
            targets.append(0)  # Cible 0 pour inversion
        else:  # Pas d'inversion, on garde les rôles d'origine
            data.at[i, 'PlayerA'] = row['Winner']
            data.at[i, 'PlayerB'] = row['Loser']
            data.at[i, 'RankA'] = row['WRank']
            data.at[i, 'RankB'] = row['LRank']
            data.at[i, 'SetsA'] = row['Wsets']
            data.at[i, 'SetsB'] = row['Lsets']
            data.at[i, 'EloA'] = row['elo_winner']
            data.at[i, 'EloB'] = row['elo_loser']
            targets.append(1)  # Cible 1 pour pas d'inversion

    # Vérifie si la longueur de 'targets' est égale à la longueur du DataFrame
    if len(targets) != len(data):
        raise ValueError(f"La longueur de 'targets' ({len(targets)}) ne correspond pas à la longueur du DataFrame ({len(data)})")

    # Ajout de la colonne 'Target' au DataFrame existant
    data['Target'] = targets

# Appliquer la fonction à 'atp' directement
create_target_with_random_inversion(atp, seed=42)

# Vérification de la structure du DataFrame modifié
display(atp.head())
display(atp.info())


# %% [markdown]
# Rempli la colonne Target avec des 1 (si PlayerA est le gagnant) ou 0 (si c’est l’inversé).
# 
# Par exemple :
# 
# Index 0 : PlayerA = Ljubicic I., PlayerB = Dosedel S., Target = 0 → cela signifie que Dosedel a gagné, mais le rôle a été inversé, donc Target = 0.
# 
# Index 1 : PlayerA = Kiefer N., PlayerB = Tarango J., Target = 1 → pas d'inversion, Kiefer a bien gagné.
# 
# 

# %%
# Extraire l'année de la colonne Date
atp['Year'] = atp['Date'].dt.year

from sklearn.preprocessing import StandardScaler
import pandas as pd

# colonnes ajoutée
atp['Rank_Diff'] = atp['RankB'] - atp['RankA']
atp['Elo_Diff'] = atp['EloB'] - atp['EloA']

# Afficher le DataFrame
display(atp.dtypes)
atp.shape

# %% [markdown]
# 🎯 Rappel : ce qu’on veut
# Dans notre dataset, le joueur A est le joueur de référence (la ligne est construite autour de lui).
# La variable cible Target = 1 signifie :
# 
# Le joueur A a gagné le match.
# Et Target = 0 veut dire :
# Le joueur A a perdu.
# 
# Donc, on veut que nos features soient cohérentes avec cette cible, pour aider le modèle à repérer les patterns entre les différences et l’issue du match.
# 
# 📊 Que représente RankB - RankA ?
# C’est :
# 
# python
# Copier
# Modifier
# Classement de l'adversaire - Classement du joueur A
# Et comme un classement ATP plus petit est meilleur (1er est mieux que 50e), alors :
# 
# Si RankB - RankA > 0 → le joueur A a un meilleur classement que son adversaire.
# 
# Si RankB - RankA < 0 → le joueur B (l’adversaire) est mieux classé que le joueur A.
# 
# Exemple :
# Joueur A	Joueur B	RankA	RankB	RankB - RankA	Interprétation
# Nadal	Schwartzman	2	12	10	Nadal est mieux classé
# Monfils	Djokovic	14	1	-13	Djokovic est mieux classé
# ✅ Pourquoi c’est mieux ?
# Parce que c’est aligné avec la cible :
# → Si RankB - RankA est positif, c’est souvent corrélé avec une victoire du joueur A.
# → Si c’est négatif, souvent corrélé avec une défaite.
# 
# 🧠 En bonus : logique inversée ?
# Tu pourrais aussi faire RankA - RankB, mais :
# 
# Le sens serait inversé (positif = A est moins bien classé)
# 
# Le modèle apprendrait pareil, mais les coefficients seraient inversés
# 
# Donc RankB - RankA est juste plus intuitif à lire (et à debug !).

# %%
###---- suppressions des colonnes unitules --
atp = atp.drop(['RankA','RankB',
               'Date','PlayerA','PlayerB','SetsA','SetsB','EloA','EloB','PSW', 'PSL','ATP','WRank','LRank',
               'proba_elo', 'elo_winner', 'elo_loser','Winner','Loser','Wsets','Lsets',], axis=1)
display(atp.dtypes)
atp.shape

# %%
atp['Year'] = pd.to_datetime(atp['Year'], format='%Y')
# Extraire l'année depuis la colonne de type datetime64
atp['Year'] = atp['Year'].dt.year

display(atp.dtypes)

# %%
# Sélectionner les lignes où Rank_Diff' = 0 
rows_to_delete = atp[(atp['Rank_Diff'] == 0) ]
# Afficher les lignes sélectionnées
display(rows_to_delete )

# %%
# Supprimer les lignes où 'Rank_Diff' = 0
atp.drop(atp[(atp['Rank_Diff'] == 0) ].index, inplace=True)
# Afficher les dimensions du DataFrame après suppression
atp.shape

# %%
# Séparer les variables explicatives (X) et la variable cible (y)
X = atp.drop(columns=['Target'])
y = atp['Target']

# Filtrer selon les années
X_train = X[X['Year'] <= 2011]   # Train : 2000 à 2011 inclus
X_test = X[X['Year'] >= 2012]    # Test : 2012 à 2018 inclus

# Aligner y avec les index de X
y_train = y.loc[X_train.index]
y_test = y.loc[X_test.index]

# Identifier les colonnes numériques et catégorielles
num_cols = X.select_dtypes(include=['float64', 'int']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Séparer les données numériques et catégorielles
num_train = X_train[num_cols]
num_test = X_test[num_cols]

cat_train = X_train[cat_cols]
cat_test = X_test[cat_cols]

# Affichage pour vérification
display(X_train)
display(X_test)


# %%
print(f"Taille du train : {len(X_train)} observations")
print(f"Taille du test  : {len(X_test)} observations")
print(f"Proportion test : {len(X_test) / (len(X_train) + len(X_test)):.2%}")
# Vérifie les plages d'années dans X_train et X_test
print("Train de", X_train['Year'].min(), "à", X_train['Year'].max())
print("Test  de", X_test['Year'].min(), "à", X_test['Year'].max())

# %% [markdown]
# Dans le cas du train_test_split pour des séries temporelles ou des données basées sur des dates, il est important de ne pas mélanger les données de manière aléatoire (ce que fait généralement shuffle=True dans train_test_split) car cela pourrait entraîner des fuites d'informations temporelles (c'est-à-dire, des événements futurs étant utilisés pour prédire le passé).
# 
# Pour traiter correctement les séries temporelles tout en séparant en ensembles d'entraînement et de test, il est recommandé de ne pas mélanger les données et de les diviser de manière chronologique, en utilisant shuffle=False.

# %%
# Gestion des NAN
from sklearn.impute import SimpleImputer

# Créer les imputers
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Appliquer les imputers
num_train = pd.DataFrame(num_imputer.fit_transform(num_train), columns=num_cols)
num_test = pd.DataFrame(num_imputer.transform(num_test), columns=num_cols)

cat_train = pd.DataFrame(cat_imputer.fit_transform(cat_train), columns=cat_cols)
cat_test = pd.DataFrame(cat_imputer.transform(cat_test), columns=cat_cols)

display(cat_train)

display(num_train)

# %%
# recherche du % de NAN ou valeurs manquantes dans atp -- vérifications
manquants_atp=cat_train.isna().mean()*100
display(manquants_atp)

cat_train.nunique()

# %%
# Encodage des variables catégorielles
from sklearn.preprocessing import OneHotEncoder

# Créer l'encodeur OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=True, handle_unknown='ignore')

# Encoder les variables catégorielles
cat_train_encoded = encoder.fit_transform(cat_train)
cat_test_encoded = encoder.transform(cat_test)

 #Créer les noms des colonnes après encodage
columns = encoder.get_feature_names_out(input_features=cat_train.columns)

# Convertir la matrice creuse en DataFrame sans la rendre dense
cat_train_encoded_df = pd.DataFrame.sparse.from_spmatrix(cat_train_encoded, columns=columns)
cat_test_encoded_df = pd.DataFrame.sparse.from_spmatrix(cat_test_encoded, columns=columns)

# Affichage du résultat
display(cat_train_encoded_df.dtypes)
display(cat_test_encoded_df)


# %%
# Vérifier le nombre de lignes et de colonnes après encodage
print(cat_train_encoded_df.shape)
print(cat_test_encoded_df.shape)

# %%
# Vérifications suite à l'encodage -------------
import pandas as pd

# Exemple de DataFrames d'entraînement et de test (cat_train et cat_test)
# cat_train et cat_test sont des DataFrames avec des colonnes catégorielles

# Identifier les catégories uniques dans l'ensemble d'entraînement et de test pour chaque colonne
for column in cat_train.columns:
    # Récupérer les catégories uniques dans cat_train et cat_test
    train_categories = set(cat_train[column].unique())
    test_categories = set(cat_test[column].unique())
    
    # Identifier les nouvelles catégories qui sont dans cat_test mais pas dans cat_train
    new_categories = test_categories - train_categories
    
    # Afficher les nouvelles catégories pour cette colonne
    if new_categories:
        print(f"Nouvelles catégories dans la colonne '{column}': {new_categories}")
    else:
        print(f"Aucune nouvelle catégorie dans la colonne '{column}'.")


# %%
# Reconstituer les jeux de données
X_train_final = pd.concat([num_train, cat_train_encoded_df], axis=1)
X_test_final = pd.concat([num_test, cat_test_encoded_df], axis=1)

# Afficher les dimensions des jeux de données
print(X_train_final.shape)
print(X_test_final.shape)
display(X_test_final.dtypes)  # Vérifiez les premières lignes du DataFrame pour voir si la colonne 'Year' existe
display(X_train_final.dtypes)

# %%
from sklearn.preprocessing import StandardScaler

# Créer le standardiseur
scaler = StandardScaler()

# Standardiser les variables numériques dans l'ensemble d'entraînement et de test
num_train_scaled = scaler.fit_transform(num_train)
num_test_scaled = scaler.transform(num_test)

# Convertir les tableaux numpy en DataFrame
num_train_scaled_df = pd.DataFrame(num_train_scaled, columns=num_train.columns)
num_test_scaled_df = pd.DataFrame(num_test_scaled, columns=num_test.columns)

# Concaténer les variables numériques standardisées avec les variables catégorielles encodées
X_train_final = pd.concat([num_train_scaled_df, cat_train_encoded_df], axis=1)
X_test_final = pd.concat([num_test_scaled_df, cat_test_encoded_df], axis=1)

# Vérification des dimensions du jeu de données après standardisation
display(X_train_final.shape)
display(X_test_final.shape)
display(X_train_final.head())
display(X_test_final.head())
display(X_train_final.dtypes)
display(X_test_final.dtypes)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Afficher un histogramme de la répartition des classes dans y_train
plt.figure(figsize=(8, 5))
sns.countplot(x=y_train)
plt.title("Répartition des classes dans y_train")
plt.xlabel("Classes")
plt.ylabel("Nombre d'exemples")
plt.show()

# %%
# Nombre d'exemples par classe dans y_train
class_counts_train = y_train.value_counts()
print("Répartition des classes dans y_train:")
print(class_counts_train)

# Nombre d'exemples par classe dans y_test
class_counts_test = y_test.value_counts()
print("Répartition des classes dans y_test:")
print(class_counts_test)

# %%
#modelisation-----------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier  # Import de k-NN

# Choisir des modèles

#rf_model = RandomForestClassifier(random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)


lr_model = LogisticRegression(max_iter=1000,random_state=42)


#dtc_model = tree.DecisionTreeClassifier(random_state=42)
dtc_model = tree.DecisionTreeClassifier(random_state=42, max_depth=10)  # Limiter la profondeur de l'arbre

#knn_model = KNeighborsClassifier(n_neighbors=5)  # k-NN avec 5 voisins
#knn_model = KNeighborsClassifier(n_neighbors=11)
knn_model = KNeighborsClassifier(n_neighbors=20)


# Entraîner les modèles
rf_model.fit(X_train_final, y_train)
lr_model.fit(X_train_final, y_train)
dtc_model.fit(X_train_final, y_train)
knn_model.fit(X_train_final, y_train)  # Entraîner k-NN


# %%
from sklearn.metrics import classification_report
import pandas as pd

# Prédictions des modèles
rf_preds = rf_model.predict(X_test_final)
lr_preds = lr_model.predict(X_test_final)
dtc_preds = dtc_model.predict(X_test_final)
knn_preds = knn_model.predict(X_test_final)  # Prédictions avec k-NN

# Comparer les performances
# Afficher la matrice de confusion pour chaque modèle
print("\nMatrice de confusion (Random Forest):")
display(pd.crosstab(y_test, rf_preds, rownames=['Réel'], colnames=['Prédiction']))

print("\nMatrice de confusion (Logistic Regression):")
display(pd.crosstab(y_test, lr_preds, rownames=['Réel'], colnames=['Prédiction']))

print("\nMatrice de confusion (Decision Tree Classifier):")
display(pd.crosstab(y_test, dtc_preds, rownames=['Réel'], colnames=['Prédiction']))

print("\nMatrice de confusion (k-NN):")
display(pd.crosstab(y_test, knn_preds, rownames=['Réel'], colnames=['Prédiction']))

# Performance des modèles
print("\nRandom Forest Performance:")
print(classification_report(y_test, rf_preds))

print("\nLogistic Regression Performance:")
print(classification_report(y_test, lr_preds))

print("\nDecision Tree Classifier Performance:")
print(classification_report(y_test, dtc_preds))

print("\nk-NN Performance:")
print(classification_report(y_test, knn_preds))

# Score sur l'ensemble de test pour chaque modèle
rf_test_score = rf_model.score(X_test_final, y_test)  # Calculer le score sur l'ensemble de test pour Random Forest
print(f'\nScore sur l\'ensemble de test (Random Forest) : {rf_test_score:.2f}')

lr_test_score = lr_model.score(X_test_final, y_test)  # Calculer le score pour Logistic Regression
print(f'Score sur l\'ensemble de test (Logistic Regression) : {lr_test_score:.2f}')

dtc_test_score = dtc_model.score(X_test_final, y_test)  # Calculer le score pour DecisionTreeClassifier
print(f'Score sur l\'ensemble de test (Decision Tree Classifier) : {dtc_test_score:.2f}')

knn_test_score = knn_model.score(X_test_final, y_test)  # Calculer le score pour knn
print(f'Score sur l\'ensemble de test (k-NN) : {knn_test_score:.2f}')


# %%
#Vérifications du sur-apprentissage ------------
from sklearn.metrics import accuracy_score

# Prédictions sur les données d'entraînement
rf_train_preds = rf_model.predict(X_train_final)
lr_train_preds = lr_model.predict(X_train_final)
dtc_train_preds = dtc_model.predict(X_train_final)
knn_train_preds = knn_model.predict(X_train_final)

# Calcul des scores (accuracy) pour chaque modèle sur les données d'entraînement et de test
rf_train_score = accuracy_score(y_train, rf_train_preds)
rf_test_score = accuracy_score(y_test, rf_preds)

lr_train_score = accuracy_score(y_train, lr_train_preds)
lr_test_score = accuracy_score(y_test, lr_preds)

dtc_train_score = accuracy_score(y_train, dtc_train_preds)
dtc_test_score = accuracy_score(y_test, dtc_preds)

knn_train_score = accuracy_score(y_train, knn_train_preds)
knn_test_score = accuracy_score(y_test, knn_preds)

# Affichage des scores
print(f"Random Forest - Train Score: {rf_train_score:.2f}, Test Score: {rf_test_score:.2f}")
print(f"Logistic Regression - Train Score: {lr_train_score:.2f}, Test Score: {lr_test_score:.2f}")
print(f"Decision Tree - Train Score: {dtc_train_score:.2f}, Test Score: {dtc_test_score:.2f}")
print(f"k-NN - Train Score: {knn_train_score:.2f}, Test Score: {knn_test_score:.2f}")


# %% [markdown]
# Si la train_accuracy est beaucoup plus élevée que la test_accuracy, il est probable que le modèle souffre de sur-apprentissage.
# 

# %%
# Créer une copie de X_test_final
test_data = X_test_final.copy()

# Réinitialiser les index pour aligner avec y_test et les prédictions
test_data = test_data.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Ajouter la colonne cible
test_data['Target'] = y_test

# Ajouter les colonnes de prédictions des modèles
test_data['RF_Pred'] = rf_preds
test_data['LR_Pred'] = lr_preds
test_data['DTC_Pred'] = dtc_preds
test_data['KNN_Pred'] = knn_preds

# Définir les favoris et les outsiders en fonction des cotes B365W et B365L
test_data['Favorite'] = test_data['B365W'] < test_data['B365L']  # Player A est favori si B365W < B365L
test_data['Outsider'] = ~test_data['Favorite']  # Le reste est outsider

# Appliquer les calculs de précision en fonction des nouvelles définitions
from sklearn.metrics import accuracy_score
results = []
for model_name in ['RF', 'LR', 'DTC', 'KNN']:
    pred_col = f'{model_name}_Pred'
    
    fav_data = test_data[test_data['Favorite']]
    fav_accuracy = accuracy_score(fav_data['Target'], fav_data[pred_col])
    
    out_data = test_data[test_data['Outsider']]
    out_accuracy = accuracy_score(out_data['Target'], out_data[pred_col])
    
    results.append({
        'Model': model_name,
        'Favori Accuracy': fav_accuracy,
        'Outsider Accuracy': out_accuracy
    })

# Résultats sous forme de DataFrame
model_performance = pd.DataFrame(results)

# Affichage
print(model_performance)

# Visualisation
import matplotlib.pyplot as plt
model_performance.set_index('Model')[['Favori Accuracy', 'Outsider Accuracy']].plot(kind='bar', figsize=(10,6))
plt.title('Performance des Modèles sur les Favoris et Outsiders')
plt.ylabel('Accuracy')
plt.xlabel('Modèle')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# %% [markdown]
# Train de 2000 à 2011
# 
# Test  de 2012 à 2018

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fonction pour simuler la stratégie Outsider
def simulate_strategy_train(X_train_final, y_train, models_dict, min_agree=3, min_odds=0, bet_amount=1, plot_title=None):
    data = X_train_final.copy().reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    data['Target'] = y_train

    # Identifier favori et outsider
    data['Favori'] = np.where(data['B365W'] < data['B365L'], 'A', 'B')
    data['Outsider'] = np.where(data['Favori'] == 'A', 'B', 'A')
    data['Outsider_Correct'] = np.where(
        (data['Target'] == 0) & (data['Outsider'] == 'B'), True, False
    )

    # Prédictions des modèles
    for name, model in models_dict.items():
        data[f'{name}_Pred'] = model.predict(X_train_final)

    pred_cols = [f'{name}_Pred' for name in models_dict]
    data['Outsider_Pred'] = data[pred_cols].mode(axis=1)[0]

    # Cas où outsider a gagné
    outsider_data = data[data['Target'] == 0].copy()
    outsider_data['Vote_0'] = (outsider_data[pred_cols] == 0).sum(axis=1)

    # Filtrer selon le niveau de confiance et la cote minimale
    qualified_bets = outsider_data[outsider_data['Vote_0'] >= min_agree].copy()
    qualified_bets['Odds'] = qualified_bets.apply(
        lambda row: row['B365W'] if row['Outsider'] == 'A' else row['B365L'], axis=1
    )
    qualified_bets = qualified_bets[qualified_bets['Odds'] >= min_odds]
    qualified_bets['Bet_Amount'] = bet_amount

    # Calcul du gain
    qualified_bets['Gain'] = qualified_bets.apply(
        lambda row: (row['Bet_Amount'] * row['Odds']) - row['Bet_Amount']
        if row['Outsider_Pred'] == row['Target']
        else -row['Bet_Amount'],
        axis=1
    )

    # Résultats globaux
    total_bet = qualified_bets['Bet_Amount'].sum()
    total_gain = qualified_bets['Gain'].sum()
    roi = (total_gain / total_bet) * 100 if total_bet > 0 else 0

    print("===== Résultats stratégie Outsider - Données d'Entraînement =====")
    print(f"Confiance : {min_agree}/4 modèles (≥ {int(min_agree/len(models_dict)*100)}%)")
    print(f"Filtre cote min. : {min_odds}")
    print(f"Nombre de paris     : {len(qualified_bets)}")
    print(f"Mise totale         : {total_bet:.2f} €")
    print(f"Gains totaux        : {total_gain:.2f} €")
    print(f"ROI                 : {roi:.2f} %")
    print(f"Qualité des cotes   : {(data['Outsider_Correct'].sum() / len(data)) * 100:.2f} %")

    if not qualified_bets.empty:
        qualified_bets['Cumulative_Gain'] = qualified_bets['Gain'].cumsum()
        qualified_bets['Cumulative_Gain'].plot(
            title=plot_title or 'Gains Cumulés - Stratégie Outsider (Entraînement)',
            figsize=(10, 6), grid=True
        )
        plt.xlabel('Pari n°')
        plt.ylabel('Gain Cumulé (€)')
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Aucun match ne remplit les conditions.")

    return qualified_bets


# Définir les modèles
models = {
    'RF': rf_model,
    'LR': lr_model,
    'DTC': dtc_model,
    'KNN': knn_model
}

# Simulation pour les données d'entraînement
simulate_strategy_train(X_train_final, y_train, models, min_agree=4, min_odds=0, bet_amount=1, plot_title="Entraînement - 100% Confiance")
simulate_strategy_train(X_train_final, y_train, models, min_agree=3, min_odds=2.5, bet_amount=1, plot_title="Entraînement - 75% Confiance + Cote ≥ 2.5")


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fonction pour simuler la stratégie Outsider
def simulate_strategy_test(X_test_final, y_test, models_dict, min_agree=3, min_odds=0, bet_amount=1, plot_title=None):
    data = X_test_final.copy().reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    data['Target'] = y_test

    # Identifier favori et outsider
    data['Favori'] = np.where(data['B365W'] < data['B365L'], 'A', 'B')
    data['Outsider'] = np.where(data['Favori'] == 'A', 'B', 'A')
    data['Outsider_Correct'] = np.where(
        (data['Target'] == 0) & (data['Outsider'] == 'B'), True, False
    )

    # Prédictions des modèles
    for name, model in models_dict.items():
        data[f'{name}_Pred'] = model.predict(X_test_final)

    pred_cols = [f'{name}_Pred' for name in models_dict]
    data['Outsider_Pred'] = data[pred_cols].mode(axis=1)[0]

    # Cas où outsider a gagné
    outsider_data = data[data['Target'] == 0].copy()
    outsider_data['Vote_0'] = (outsider_data[pred_cols] == 0).sum(axis=1)

    # Filtrer selon le niveau de confiance et la cote minimale
    qualified_bets = outsider_data[outsider_data['Vote_0'] >= min_agree].copy()
    qualified_bets['Odds'] = qualified_bets.apply(
        lambda row: row['B365W'] if row['Outsider'] == 'A' else row['B365L'], axis=1
    )
    qualified_bets = qualified_bets[qualified_bets['Odds'] >= min_odds]
    qualified_bets['Bet_Amount'] = bet_amount

    # Calcul du gain
    qualified_bets['Gain'] = qualified_bets.apply(
        lambda row: (row['Bet_Amount'] * row['Odds']) - row['Bet_Amount']
        if row['Outsider_Pred'] == row['Target']
        else -row['Bet_Amount'],
        axis=1
    )

    # Résultats globaux
    total_bet = qualified_bets['Bet_Amount'].sum()
    total_gain = qualified_bets['Gain'].sum()
    roi = (total_gain / total_bet) * 100 if total_bet > 0 else 0

    print("===== Résultats stratégie Outsider - Données de Test =====")
    print(f"Confiance : {min_agree}/4 modèles (≥ {int(min_agree/len(models_dict)*100)}%)")
    print(f"Filtre cote min. : {min_odds}")
    print(f"Nombre de paris     : {len(qualified_bets)}")
    print(f"Mise totale         : {total_bet:.2f} €")
    print(f"Gains totaux        : {total_gain:.2f} €")
    print(f"ROI                 : {roi:.2f} %")
    print(f"Qualité des cotes   : {(data['Outsider_Correct'].sum() / len(data)) * 100:.2f} %")

    if not qualified_bets.empty:
        qualified_bets['Cumulative_Gain'] = qualified_bets['Gain'].cumsum()
        qualified_bets['Cumulative_Gain'].plot(
            title=plot_title or 'Gains Cumulés - Stratégie Outsider (Test)',
            figsize=(10, 6), grid=True
        )
        plt.xlabel('Pari n°')
        plt.ylabel('Gain Cumulé (€)')
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Aucun match ne remplit les conditions.")

    return qualified_bets


# Définir les modèles
models = {
    'RF': rf_model,
    'LR': lr_model,
    'DTC': dtc_model,
    'KNN': knn_model
}

# Simulation pour les données de test
simulate_strategy_test(X_test_final, y_test, models, min_agree=4, min_odds=0, bet_amount=1, plot_title="Test - 100% Confiance")
simulate_strategy_test(X_test_final, y_test, models, min_agree=3, min_odds=2.5, bet_amount=1, plot_title="Test - 75% Confiance + Cote ≥ 2.5")



