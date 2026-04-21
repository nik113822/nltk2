# import numpy as np
# from sklearn.naive_bayes import GaussianNB


# # 1. Τα δεδομένα μας (Features: [Ύψος, Βάρος, Παπούτσι])
# # Χ = Χαρακτηριστικά
# X = np.array([
#     [183, 85, 44],  # Άνδρας
#     [180, 80, 43],  # Άνδρας
#     [170, 75, 42],  # Άνδρας
#     [160, 50, 37],  # Γυναίκα
#     [165, 55, 38],  # Γυναίκα
#     [155, 45, 36]   # Γυναίκα
# ])

# # y = Ετικέτες (Labels: 'male' ή 'female')
# y = np.array([
#     'male', 'male', 'male',
#     'female', 'female', 'female'
# ])

# # 2. Δημιουργία του Μοντέλου (Ο "εγκέφαλος")
# clf = GaussianNB()

# # 3. Εκπαίδευση (Training)
# # Εδώ ο αλγόριθμος μαθαίνει τη σχέση μεταξύ αριθμών και φύλου
# clf.fit(X, y)

# # 4. Πρόβλεψη (Prediction)
# # Ας του δώσουμε ένα νέο άτομο που δεν έχει ξαναδεί:
# # Ύψος: 178, Βάρος: 78, Παπούτσι: 41
# neo_atomo = [[178, 78, 41]]

# prediction = clf.predict(neo_atomo)

# print(f"Το μοντέλο προβλέπει ότι το άτομο είναι {prediction[0]}")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Εισάγουμε και τα 3 είδη Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# 1. Τα Δεδομένα μας
X = np.array([
    [183, 85, 44], [180, 80, 43], [170, 75, 42], [175, 78, 42], [185, 90, 45], 
    [160, 50, 37], [165, 55, 38], [155, 45, 36], [158, 48, 37], [162, 52, 38] 
])
y = np.array([
    'male', 'male', 'male', 'male', 'male', 
    'female', 'female', 'female', 'female', 'female'
])

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Δημιουργία Λίστας με τα Μοντέλα ---
models = {
    "Gaussian": GaussianNB(),
    "Multinomial": MultinomialNB(),
    "Bernoulli": BernoulliNB()
}

print(f"Πραγματικές τιμές (Test): {y_test}\n")

# 3. Τρέχουμε έναν βρόχο (loop) για να τα δοκιμάσουμε όλα
for name, clf in models.items():
    # Εκπαίδευση
    clf.fit(X_train, y_train)
    
    # Πρόβλεψη
    prediction = clf.predict(X_test)
    
    # Αξιολόγηση
    acc = accuracy_score(y_test, prediction) * 100
    
    print(f"--- Μοντέλο: {name} NB ---")
    print(f"Πρόβλεψη: {prediction}")
    print(f"Ακρίβεια: {acc}%\n")