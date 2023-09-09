import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Carga de datos y entrenamiento del modelo (mantén esta parte igual)

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
data.isnull().sum()
data["language"].value_counts()
x = np.array(data["Text"])
y = np.array(data["language"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Mapeo de nombres de idiomas en inglés a español
idiomas = {
    'English': 'Inglés',
    'Spanish': 'Español',
    'Chinese': 'Chino',
    'French': 'Francés',
    'German': 'Alemán',
    'Japanese': 'Japonés',
    'Korean': 'Coreano',
    'Russian': 'Ruso',
    'Arabic': 'Árabe',
    'Italian': 'Italiano',
    'Dutch': 'Holandés',
    'Portuguese': 'Portugués',
    'Swedish': 'Sueco',
    'Turkish': 'Turco',
    'Greek': 'Griego',
    'Hindi': 'Hindi',
    'Bengali': 'Bengalí',
    'Urdu': 'Urdu',
    'Punjabi': 'Punyabí',
    'Persian': 'Persa',
    'Vietnamese': 'Vietnamita',
    'Thai': 'Tailandés',
    'Indonesian': 'Indonesio',
    'Malay': 'Malayo',
    'Swahili': 'Suajili',
}

# Bienvenida y creación del menú
print("Bienvenido al detector de idiomas")
print("")


while True:
    print("Escriba un texto para determinar el idioma o [salir] para salir ")
    print("")
    user = input("Ingrese el texto: ")
    
    if user.lower() == 'salir':
        print("¡Hasta luego!")
        break

    # Verificamos si el texto tiene menos de tres palabras
    if len(user.split()) < 3:
        print("")
        print("El texto debe contener al menos tres palabras. Intente nuevamente.")
        print("")
        continue
    
    data = cv.transform([user]).toarray()
    output = model.predict(data)
    idioma_predicho = idiomas.get(output[0], output[0])
    print("")
    print("El Idioma es : ", idioma_predicho)
    print("")




