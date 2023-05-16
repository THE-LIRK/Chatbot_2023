import streamlit as st
import nltk
import pandas as pd
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



# Définir la couleur de fond en rose
# Affichage de l'interface utilisateur
st.markdown("<h1 style='text-align: center; color: red;'>Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stApp {
        background-image:url('https://w.wallhaven.cc/full/39/wallhaven-398lo3.png');
        background-size: cover;
        color: red ;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Titre de la barre latérale
st.sidebar.title("Comment fonctionne ce projet ?")

# Texte d'explication
st.sidebar.write("Ce projet est un chatbot médical qui permet de prédire des maladies à partir de symptômes choisis par l'utilisateur.Pour utiliser le chatbot, suivez les étapes suivantes :")

# Liste d'étapes
st.sidebar.write("1.Écrivez 'Je suis malade' pour que le processus se lance.")
st.sidebar.write("2. Entrez le nombre de symptômes que vous voulez sélectionner ")
st.sidebar.write("3. Sélectionnez les symptômes dans les listes déroulantes")
st.sidebar.write("4. Cliquez sur le bouton 'Predire'")
st.sidebar.write("5. Le chatbot fera une prédiction de la maladie basée sur les symptômes que vous avez choisis")

# Texte de fin
st.sidebar.write("Si vous avez des questions, n'hésitez pas à contacter l'auteur du projet.")





# initialisation du lemmatizer de NLTK
lemmatizer = WordNetLemmatizer()





# liste des symptômes
symptoms = [
    'itching', ' skin_rash', ' continuous_sneezing', ' shivering',
       ' stomach_pain', ' acidity', ' vomiting', ' indigestion',
       ' muscle_wasting', ' patches_in_throat', ' fatigue',
       ' weight_loss', ' sunken_eyes', ' cough', ' headache',
       ' chest_pain', ' back_pain', ' weakness_in_limbs', ' chills',
       ' joint_pain', ' yellowish_skin', ' constipation',
       ' pain_during_bowel_movements', ' breathlessness', ' cramps',
       ' weight_gain', ' mood_swings', ' neck_pain', ' muscle_weakness',
       ' stiff_neck', ' pus_filled_pimples', ' burning_micturition',
       ' bladder_discomfort', ' high_fever', ' nodal_skin_eruptions',
       ' ulcers_on_tongue', ' loss_of_appetite', ' restlessness',
       ' dehydration', ' dizziness', ' weakness_of_one_body_side',
       ' lethargy', ' nausea', ' abdominal_pain', ' pain_in_anal_region',
       ' sweating', ' bruising', ' cold_hands_and_feets', ' anxiety',
       ' knee_pain', ' swelling_joints', ' blackheads',
       ' foul_smell_of urine', ' skin_peeling', ' blister',
       ' dischromic _patches', ' watering_from_eyes',
       ' extra_marital_contacts', ' diarrhoea', ' loss_of_balance',
       ' blurred_and_distorted_vision', ' altered_sensorium',
       ' dark_urine', ' swelling_of_stomach', ' bloody_stool', ' obesity',
       ' hip_joint_pain', ' movement_stiffness', ' spinning_movements',
       ' scurring', ' continuous_feel_of_urine', ' silver_like_dusting',
       ' red_sore_around_nose', ' spotting_ urination',
       ' passage_of_gases', ' irregular_sugar_level', ' family_history',
       ' lack_of_concentration', ' excessive_hunger',
       ' yellowing_of_eyes', ' distention_of_abdomen',
       ' irritation_in_anus', ' swollen_legs', ' painful_walking',
       ' small_dents_in_nails', ' yellow_crust_ooze', ' internal_itching',
       ' mucoid_sputum', ' history_of_alcohol_consumption',
       ' swollen_blood_vessels', ' unsteadiness', ' inflammatory_nails',
       ' depression', ' fluid_overload', ' swelled_lymph_nodes',
       ' malaise', ' prominent_veins_on_calf', ' puffy_face_and_eyes',
       ' fast_heart_rate', ' irritability', ' muscle_pain', ' mild_fever',
       ' yellow_urine', ' phlegm', ' enlarged_thyroid',
       ' increased_appetite', ' visual_disturbances', ' brittle_nails',
       ' drying_and_tingling_lips', ' polyuria', ' pain_behind_the_eyes',
       ' toxic_look_(typhos)', ' throat_irritation',
       ' swollen_extremeties', ' slurred_speech', ' red_spots_over_body',
       ' belly_pain', ' receiving_blood_transfusion',
       ' acute_liver_failure', ' redness_of_eyes', ' rusty_sputum',
       ' abnormal_menstruation', ' receiving_unsterile_injections',
       ' coma', ' sinus_pressure', ' palpitations'
]


#Divisions des symptones en 11 listes 
symptoms_lists = [
    ['itching', ' skin_rash', ' continuous_sneezing', ' shivering',
       ' stomach_pain', ' acidity', ' vomiting', ' indigestion',
       ' muscle_wasting', ' patches_in_throat', ' fatigue',
       ' weight_loss', ' sunken_eyes', ' cough', ' headache',
       ' chest_pain', ' back_pain', ' weakness_in_limbs', ' chills',
       ' joint_pain', ' yellowish_skin', ' constipation',
       ' pain_during_bowel_movements', ' breathlessness', ' cramps',
       ' weight_gain', ' mood_swings', ' neck_pain', ' muscle_weakness',
       ' stiff_neck', ' pus_filled_pimples', ' burning_micturition',
       ' bladder_discomfort', ' high_fever'],

    [' skin_rash', ' nodal_skin_eruptions', ' shivering', ' chills',
       ' acidity', ' ulcers_on_tongue', ' vomiting', ' yellowish_skin',
       ' stomach_pain', ' loss_of_appetite', ' indigestion',
       ' patches_in_throat', ' high_fever', ' weight_loss',
       ' restlessness', ' sunken_eyes', ' dehydration', ' cough',
       ' chest_pain', ' dizziness', ' headache', ' weakness_in_limbs',
       ' neck_pain', ' weakness_of_one_body_side', ' fatigue',
       ' joint_pain', ' lethargy', ' nausea', ' abdominal_pain',
       ' pain_during_bowel_movements', ' pain_in_anal_region',
       ' breathlessness', ' sweating', ' cramps', ' bruising',
       ' weight_gain', ' cold_hands_and_feets', ' mood_swings',
       ' anxiety', ' knee_pain', ' stiff_neck', ' swelling_joints',
       ' pus_filled_pimples', ' blackheads', ' bladder_discomfort',
       ' foul_smell_of urine', ' skin_peeling', ' blister'],

    [' nodal_skin_eruptions', ' dischromic _patches', ' chills',
       ' watering_from_eyes', ' ulcers_on_tongue', ' vomiting',
       ' yellowish_skin', ' nausea', ' stomach_pain',
       ' burning_micturition', ' abdominal_pain', ' loss_of_appetite',
       ' high_fever', ' extra_marital_contacts', ' restlessness',
       ' lethargy', ' dehydration', ' diarrhoea', ' breathlessness',
       ' dizziness', ' loss_of_balance', ' headache',
       ' blurred_and_distorted_vision', ' neck_pain',
       ' weakness_of_one_body_side', ' altered_sensorium', ' fatigue',
       ' weight_loss', ' sweating', ' joint_pain', ' dark_urine',
       ' swelling_of_stomach', ' cough', ' pain_in_anal_region',
       ' bloody_stool', ' chest_pain', ' bruising', ' obesity',
       ' cold_hands_and_feets', ' mood_swings', ' anxiety', ' knee_pain',
       ' hip_joint_pain', ' swelling_joints', ' movement_stiffness',
       ' spinning_movements', ' blackheads', ' scurring',
       ' foul_smell_of urine', ' continuous_feel_of_urine',
       ' skin_peeling', ' silver_like_dusting', ' blister',
       ' red_sore_around_nose'],

    [' dischromic _patches', 'pas', ' watering_from_eyes', ' vomiting',
       ' cough', ' nausea', ' loss_of_appetite', ' burning_micturition',
       ' spotting_ urination', ' passage_of_gases', ' abdominal_pain',
       ' extra_marital_contacts', ' lethargy', ' irregular_sugar_level',
       ' diarrhoea', ' breathlessness', ' family_history',
       ' loss_of_balance', ' lack_of_concentration',
       ' blurred_and_distorted_vision', ' excessive_hunger', ' dizziness',
       ' altered_sensorium', ' weight_loss', ' high_fever', ' sweating',
       ' headache', ' fatigue', ' dark_urine', ' yellowish_skin',
       ' yellowing_of_eyes', ' swelling_of_stomach',
       ' distention_of_abdomen', ' bloody_stool', ' irritation_in_anus',
       ' chest_pain', ' obesity', ' swollen_legs', ' mood_swings',
       ' restlessness', ' hip_joint_pain', ' swelling_joints',
       ' movement_stiffness', ' painful_walking', ' spinning_movements',
       ' scurring', ' continuous_feel_of_urine', ' silver_like_dusting',
       ' small_dents_in_nails', ' red_sore_around_nose',
       ' yellow_crust_ooze'],

    ['pas', ' cough', ' chest_pain', ' loss_of_appetite',
       ' abdominal_pain', ' spotting_ urination', ' internal_itching',
       ' passage_of_gases', ' irregular_sugar_level',
       ' blurred_and_distorted_vision', ' family_history',
       ' mucoid_sputum', ' lack_of_concentration', ' excessive_hunger',
       ' stiff_neck', ' loss_of_balance', ' high_fever',
       ' yellowish_skin', ' headache', ' nausea', ' fatigue',
       ' dark_urine', ' yellowing_of_eyes', ' distention_of_abdomen',
       ' history_of_alcohol_consumption', ' breathlessness', ' sweating',
       ' irritation_in_anus', ' swollen_legs', ' swollen_blood_vessels',
       ' lethargy', ' dizziness', ' diarrhoea', ' swelling_joints',
       ' painful_walking', ' unsteadiness', ' small_dents_in_nails',
       ' inflammatory_nails', ' yellow_crust_ooze'],

    ['pas', ' chest_pain', ' abdominal_pain', ' yellowing_of_eyes',
       ' internal_itching', ' blurred_and_distorted_vision', ' obesity',
       ' mucoid_sputum', ' stiff_neck', ' depression', ' yellowish_skin',
       ' dark_urine', ' nausea', ' diarrhoea', ' headache',
       ' loss_of_appetite', ' high_fever', ' constipation',
       ' family_history', ' history_of_alcohol_consumption',
       ' fluid_overload', ' breathlessness', ' swelled_lymph_nodes',
       ' sweating', ' malaise', ' swollen_blood_vessels',
       ' prominent_veins_on_calf', ' dizziness', ' puffy_face_and_eyes',
       ' fast_heart_rate', ' painful_walking', ' unsteadiness',
       ' inflammatory_nails'],

    ['pas', ' yellowing_of_eyes', ' obesity', ' excessive_hunger',
       ' depression', ' irritability', ' dark_urine', ' abdominal_pain',
       ' muscle_pain', ' diarrhoea', ' loss_of_appetite', ' mild_fever',
       ' headache', ' nausea', ' constipation', ' yellow_urine',
       ' fluid_overload', ' breathlessness', ' sweating',
       ' swelled_lymph_nodes', ' malaise', ' phlegm',
       ' prominent_veins_on_calf', ' puffy_face_and_eyes',
       ' enlarged_thyroid', ' fast_heart_rate',
       ' blurred_and_distorted_vision'],

    ['pas', ' excessive_hunger', ' increased_appetite', ' irritability',
       ' visual_disturbances', ' abdominal_pain', ' muscle_pain',
       ' mild_fever', ' swelled_lymph_nodes', ' nausea',
       ' loss_of_appetite', ' diarrhoea', ' yellow_urine',
       ' yellowing_of_eyes', ' sweating', ' malaise', ' phlegm',
       ' chest_pain', ' enlarged_thyroid', ' brittle_nails',
       ' muscle_weakness', ' drying_and_tingling_lips'],

    ['pas', ' increased_appetite', ' polyuria', ' visual_disturbances',
       ' swelled_lymph_nodes', ' malaise', ' loss_of_appetite',
       ' pain_behind_the_eyes', ' toxic_look_(typhos)', ' diarrhoea',
       ' mild_fever', ' yellowing_of_eyes', ' abdominal_pain', ' phlegm',
       ' throat_irritation', ' fast_heart_rate', ' chest_pain',
       ' brittle_nails', ' swollen_extremeties', ' muscle_weakness',
       ' irritability', ' slurred_speech', ' drying_and_tingling_lips'],

    ['pas', ' polyuria', ' malaise', ' red_spots_over_body',
       ' pain_behind_the_eyes', ' back_pain', ' belly_pain',
       ' toxic_look_(typhos)', ' yellowing_of_eyes', ' muscle_pain',
       ' receiving_blood_transfusion', ' acute_liver_failure',
       ' mild_fever', ' throat_irritation', ' redness_of_eyes',
       ' rusty_sputum', ' fast_heart_rate', ' swollen_extremeties',
       ' depression', ' irritability', ' abnormal_menstruation',
       ' slurred_speech'],

    ['pas', ' red_spots_over_body', ' back_pain', ' malaise',
       ' belly_pain', ' muscle_pain', ' receiving_blood_transfusion',
       ' receiving_unsterile_injections', ' coma', ' acute_liver_failure',
       ' yellowing_of_eyes', ' swelled_lymph_nodes', ' redness_of_eyes',
       ' sinus_pressure', ' rusty_sputum', ' depression', ' irritability',
       ' abnormal_menstruation', ' palpitations']
]





# importer la base de donner 
p="dataset.csv"
dt=pd.read_csv(p)
#supprimer les colonnes vides à 75%
def vide(data):
    for c in data.columns:
        if data[c].isna().sum()>4919*76/100 :
            data.drop(columns=[c],inplace=True)
vide(dt)

#le nombre de symptomes
dt["nb_symptomes"]=dt.drop(columns=["Disease"]).apply(lambda x: x.count(), axis=1)

#remplacer les nan par les pas 
dt = dt.fillna('pas')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Créer un encodeur LabelEncoder pour la variable cible y
encoder_y = LabelEncoder()
y = dt['Disease']

# Créer un encodeur OneHotEncoder pour les variables catégorielles dans les données
encoder_x = OneHotEncoder()
x = dt.drop(['Disease',"nb_symptomes"], axis=1)
encoder_x.fit(x) # ajuster l'encodeur aux données d'entraînement
x = encoder_x.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)

model = DecisionTreeClassifier(random_state=42)

#Train
model.fit(x_train, y_train)


# Fonction pour nettoyer le texte de l'utilisateur
def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if not token in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    clean_text = ' '.join(tokens)
    return clean_text

# Dictionnaire de questions et de réponses
responses = {
    "Bonjour": "Bonjour, comment puis-je vous aider?",
    "Comment allez-vous?": "Je vais bien, merci. Et vous?",
    "ça va ?":"Je vais bien, merci. Et vous?",
    "Je vais bien": "Super",
    "Quel est la fiabilite de ce modele":"Quel est la fiabilité de ce modèle ?\n"
      "Accuracy moyenne :  0.9932294539437396\n"
      "Précision moyenne :  0.9947154471544714\n"
      "Rappel moyen :  0.9928048780487805\n"
      "F1-mesure moyenne :  0.9927864005912787" ,
    "Allergy":"Une allergie est une réaction du système immunitaire à une substance étrangère appeléeallergène, qui peut provoquer des symptômes tels que des éruptions cutanées, deséternuements, des démangeaisons et des difficultés respiratoires. Le traitement dépend de la gravité des symptômes et peut inclure des médicaments et des mesures préventives",
    "Fungal infection":"Une infection fongique est une maladie causée par un champignon qui peut affecter la peau,les ongles, les cheveux, la bouche, les organes génitaux et les poumons. Les symptômes peuvent varier selon la partie affectée, mais peuvent inclure des rougeurs, des démangeaisons, des gonflements, des douleurs et des éruptions cutanées. Les médicaments antifongiques sont souvent utilisés pour traiter les infections fongiques, et les mesures préventives comprennent une bonne hygiène et l'évitement d'environnements humides.",
    "Je suis malade":"Combien de symptômes avez-vous?",
    "Non":"Combien de symptômes avez-vous?",
    "Je ne me sens pas bien":"Combien de symptômes avez-vous?",
    "Quel est votre nom?": "Je suis un chatbot. Comment puis-je vous aider?",
    "Au revoir": "Au revoir, à bientôt!",
    
    "Qu'est-ce que ce projet ?": "Ce projet est un chatbot médical qui permet de prédire des maladies à partir de symptômes choisis par l'utilisateur.",
    "Comment utiliser le chatbot ?": "Pour utiliser le chatbot, vous devez écrire 'Je suis malade' pour que le processus se lance. Ensuite, entrez le nombre de symptômes que vous voulez sélectionner et choisissez les symptômes dans les listes déroulantes. Cliquez sur le bouton 'Stocker les symptômes choisis' pour que le chatbot fasse une prédiction de la maladie basée sur les symptômes que vous avez choisis.",
    "Quel est le modèle utilisé pour faire les prédictions ?": "Le modèle utilisé est un algorithme de machine learning basé sur l'encodage one-hot des symptômes et la classification multiclasse.",
    "Comment puis-je contacter l'auteur du projet ?": "Vous pouvez contacter l'auteur du projet en envoyant un e-mail à roddylepenguet@gmail.com."
}

# Fonction pour répondre à l'utilisateur
def get_response(user_input):
    user_input = clean_text(user_input)
    for key, value in responses.items():
        if user_input == clean_text(key):
            return value
    return "Désolé, je ne comprends pas ce que vous dites."



user_input = st.text_input("Vous: ")
response = get_response(user_input)



selected_symptoms = []

if response == responses["Je suis malade"or"Je ne me sens pas bien"or"Non"]:
    # Demander le nombre de symptômes
   
    num_symptoms = st.number_input("", min_value=4, max_value=11, step=1)
    
    # Afficher les listes déroulantes pour chaque symptôme
    for i in range(num_symptoms):
        st.write(f"Veuillez sélectionner le symptôme {i+1}:")
        selected_symptom = st.selectbox("", list(symptoms_lists[i]))
        selected_symptoms.append(selected_symptom)

    # compléter la liste avec "pas" jusqu'à obtenir 11 symptômes
    while len(selected_symptoms) < 11:
        selected_symptoms.append("pas")
            
    # bouton pour stocker les symptômes choisis
    if st.button("Predire"):
        # soumettre les symptômes au modèle de machine learning
        # code à ajouter pour soumettre les symptômes au modèle de machine learning
        
        selected_symptoms=[selected_symptoms]
            
        new_data = np.array(selected_symptoms).reshape(1, -1)
        # soumettre le tableau transformé à l'encodeur OneHotEncoder
        new_data_encoded = encoder_x.transform(new_data)
        # Faire des prédictions sur les nouvelles données
        predictions = model.predict(new_data_encoded)
        st.write(f"Vous souffrez de : {predictions}")


else:
    st.text_area("Chatbot:", value=response, height=200, max_chars=None, key=None)















   
