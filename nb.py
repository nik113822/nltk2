from sklearn.naive_bayes import BernoulliNB
import numpy as np

# 1. Δεδομένα: [Έχει Σύννεφα?, Έχει Υγρασία?, Φυσάει?]
# 1 = Ναι, 0 = Όχι
X_train = np.array([
    [1, 1, 0], # Συννεφιά + Υγρασία -> Βροχή
    [0, 0, 1], # Μόνο Αέρας -> Όχι Βροχή
    [1, 0, 0], # Μόνο Συννεφιά -> Όχι Βροχή
    [1, 1, 1]  # Όλα -> Βροχή
])

y_train = np.array(['Βροχή', 'Όχι Βροχή', 'Όχι Βροχή', 'Βροχή'])

# 2. Μοντέλο Bernoulli (για binary data)
clf = BernoulliNB()
clf.fit(X_train, y_train)

# 3. Πρόβλεψη
# Σημερινή μέρα: Έχει Συννεφιά (1), Όχι Υγρασία (0), Όχι Αέρα (0)
shmera = [[1, 0, 0]]

print("Πρόβλεψη Bernoulli:", clf.predict(shmera)[0])