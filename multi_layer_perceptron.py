import gzip # pour décompresser les données
import pickle
import numpy # pour pouvoir utiliser des matrices
import pylab # pour l'affichage
import copy
import random
import time

#pour rendre l'affichage plus lisible en supprimant l'affichage "scientifique"
#numpy.set_printoptions(suppress=True)

#paramètrage
nbStep = 125000; #nombre d'itérations
nbLayer = 3 ; #nombre de couches dans le réseau de neurones (en comptant la couche de sortie)
nbNeurons = 150; #nombre de neurones par couche
learningRate = 0.1; #taux d'apprentissage

#Affichage des paramètres
print("nb couches : " + str(nbLayer) + " nb neurones : " + str(nbNeurons))

#chargement de la base de données.
print ("Chargement de la base de données...")
f = gzip.open('mnist.pkl.gz')
data = pickle.load(f,encoding='latin1')
f04 = gzip.open('mnist0-4.pkl.gz')
data04 = pickle.load(f04,encoding='latin1')
f59 = gzip.open('mnist5-9.pkl.gz')
data59 = pickle.load(f59,encoding='latin1')
print("Base de données chargée.")
#determination de la taille des vecteurs d'entrées et de sorties
inputSize = len(data[0][0][0]) + 1
outputSize = len(data[0][1][0])

#Les poids définissant le réseau sont stockés dans une liste de matices de poids. Chaque matrice représente l'ensemble de poids
# présent entre 2 couches. Une ligne de la matrice contient les poids associés à un neurone cible
# et chaque colonne contient les poids associés à un neurone source
def init (weightML) :
    if (nbLayer == 1) : #dans le cas où il n'y a pas de couche cachée
        weightML.append(numpy.zeros((outputSize,inputSize)))
        for j in  range(0, outputSize) :
            for k in range(0, inputSize) :
                weightML[0][j][k] = random.uniform(-1,1) #asination d'un nombre aléatoire entre -1 et 1
    else :
        for i in range (0, nbLayer) :
            if i == 0 : # premier ensemble de poids, entre entrée et première couche
                weightML.append(numpy.zeros((nbNeurons,inputSize)))
                for j in  range(0, nbNeurons) :
                    for k in range(0, inputSize) :
                        weightML[i][j][k] = random.uniform(-1,1)
            elif i != nbLayer-1 : #ensemble de poids entre 2 couches cachées
                weightML.append(numpy.zeros((nbNeurons,nbNeurons + 1)))
                for j in  range(0, nbNeurons) :
                    for k in range(0, nbNeurons) :
                        weightML[i][j][k] = random.uniform(-1,1)
            else : # ensemble de poids situé entre la dernière couche cachée et la sortie
                weightML.append(numpy.zeros((outputSize,nbNeurons + 1)))
                for j in  range(0, outputSize) :
                    for k in range(0, nbNeurons+1) :
                        weightML[i][j][k] = random.uniform(-1,1)

def Apprentissage (weightML, base_app) :
    #déclaration des structures qui vont stocker les valeurs des neurones
    out = numpy.empty((0,0))
    tmp = numpy.empty((0,0))

    print ("Début de l'apprentissage...")

    for i in range (0,nbStep) : # sur n pas de temps
        #selection d'une entrée au hasard dans la base d'apprentissage
        randomInt = random.randint(0,len(base_app[0][0])-1)
        vec = base_app[0][0][randomInt]
        vec = numpy.append(vec, 1) #ajout du biais
        neuronsVL = [] # création de la structure qui stocke les valeurs des neurones pour un pas de temps

        #forward
        for j in range (0,nbLayer) : #pour chaque couche du réseau application de la fonction d'activation
            if j == 0 : #si on est sur la première couche, on prend l'entrée vec
                tmp = numpy.dot(weightML[j], vec) #le produit matriciel remplace la somme
            else : #sinon, on prend les données calculées dans la couche précedente
                tmp = numpy.dot(weightML[j], neuronsVL[j-1]) #le produit matriciel remplace la somme

            tmp = (1/(1 + numpy.exp(-tmp)))

            if j == nbLayer-1 : #si on est sur la dernière couche, on est sur la sortie -> pas d'ajout de biais, mise dans out
                out = tmp
            else : #attention au biais -> ajouter 1 à la fin du tableau si ce n'est pas la dernière couche
                neuronsVL.append(numpy.append(tmp,1))

        #backward
        for l in reversed (range(0,nbLayer)) :
            if l == nbLayer-1 :
                #calcul de l'erreur pour la couche de sortie
                err = numpy.empty(out.shape)
                err = out*(1-out)*(base_app[0][1][randomInt]-out)

                #correction des poids liés à la couche de sortie
                weight_old = copy.deepcopy(weightML[l])
                err_old = copy.deepcopy(err)
                #pour chaque poids, on fait dans une nouvelle matrice le produit (erreur, valeur du neurone à la couche précédente)
                if l != 0 :
                    delta = numpy.dot(numpy.reshape(err,(err.size,1)), numpy.transpose(numpy.reshape(neuronsVL[l-1],(neuronsVL[l-1].size,1))))
                    weightML[l] = weightML[l] + (delta*learningRate)

            else :
                #retropropagation de l'erreur
                err = numpy.empty((neuronsVL[l].size-1,1))
                for k in range (0,neuronsVL[l].size-1) :
                    #la somme présente dans le calcul de l'erreur propagée est a nouveau remplacé par un produit matriciel
                    som = numpy.dot(numpy.transpose(numpy.reshape(weight_old[:,k],(weight_old[:,k].size,1))),numpy.reshape(err_old,(err_old.size,1)))
                    err[k] = neuronsVL[l][k] * (1 - neuronsVL[l][k]) * som
                #correction poids
                if l != 0 :
                    weight_old = copy.deepcopy(weightML[l])
                    err_old = copy.deepcopy(err)
                    delta = numpy.dot(numpy.reshape(err,(err.size,1)), numpy.transpose(numpy.reshape(neuronsVL[l-1],(neuronsVL[l-1].size,1))))
                    weightML[l] = weightML[l] + (delta*learningRate)
                else : # l == 0
                    delta = numpy.dot(numpy.reshape(err,(err.size,1)), numpy.transpose(numpy.reshape(vec,(vec.size,1))))
                    weightML[l] = weightML[l] + (delta*learningRate)
    print ("Apprentissage terminé")

def test(res, base_test) :
    print ("Début de la phase de test...")
    out = numpy.empty((0,0))
    tmp = numpy.empty((0,0))
    bonnes_reponses = 0
    reponses = 0
    for i in range (0,len(base_test[1][0])) : # pour toutes les entrées présentes dans la base de test
        vec = base_test[1][0][i]
        vec = numpy.append(vec, 1)
        neuronsVL = []
        for j in range (0,nbLayer) : #propagation dans toutes les couches du réseau
                if j == 0 :
                    tmp = numpy.dot(res[j], vec)
                else :
                    tmp = numpy.dot(res[j], neuronsVL[j-1])

                tmp = (1/(1 + numpy.exp(-tmp)))

                if j == nbLayer-1 :
                    out = tmp
                else :
                    neuronsVL.append(numpy.append(tmp,1))
        reponses = reponses + 1
        if numpy.argmax(out) == numpy.argmax(base_test[1][1][i]) : #comparaison avec la sortie attendue
            bonnes_reponses = bonnes_reponses + 1 #si le résultat produit est bon on augmente le nombre de bonnes réponses

    print ("Taux de bonnes réponses : " + str(bonnes_reponses/reponses*100)) #calcul du taux de réussite
    print ("Phase de test terminée.")


# Code Question 2---------------------------------------------------------
wml = []
init(wml);
t1=time.process_time()
Apprentissage(wml, data)
t2=time.process_time()
print ("Apprentissage terminé en : " + str(t2 - t1) + " secondes")
test(wml, data)

 # Code Question 3---------------------------------------------------------
 #entrainement des 3 réseaux
# print("Intialisation réseau All")
# wmlAll = []
# init(wmlAll);
# print("Apprentissage réseau All")
# Apprentissage(wmlAll, data)
#
# print("Intialisation réseau 04")
# wml04 = []
# init(wml04);
# print("Apprentissage réseau 04")
# Apprentissage(wml04, data04)
#
# print("Intialisation réseau 59")
# wml59 = []
# init(wml59);
# print("Apprentissage réseau 59")
# Apprentissage(wml59, data59)
#
# #test
#
# #test réseau All
# print ("Test Réseau All sur base complète")
# test(wmlAll, data)
# print ("Test Réseau All sur base 04")
# test(wmlAll, data04)
# print ("Test Réseau All sur base 59")
# test(wmlAll, data59)
#
# #test réseau 04
# print ("Test Réseau 04 sur base complète")
# test(wml04, data)
# print ("Test Réseau 04 sur base 04")
# test(wml04, data04)
# print ("Test Réseau 04 sur base 59")
# test(wml04, data59)
#
# #test réseau 59
# print ("Test Réseau 59 sur base complète")
# test(wml59, data)
# print ("Test Réseau 59 sur base 04")
# test(wml59, data04)
# print ("Test Réseau 59 sur base 59")
# test(wml59, data59)

# Code Question 3---------------------------------------------------------
# print("Intialisation réseau 04")
# wml04 = []
# init(wml04);
# print("Apprentissage réseau 04 sur base 04")
# Apprentissage(wml04, data04)
#
# print ("Test 1 Réseau 59 sur base 04")
# test(wml04, data04)
# print ("Test 1 Réseau 59 sur base 59")
# test(wml04, data59)
#
# print("Apprentissage réseau 04 sur base 04")
# Apprentissage(wml04, data59)
#
# print ("Test 2 Réseau 04 sur base 04")
# test(wml04, data04)
# print ("Test 2 Réseau 04 sur base 59")
# test(wml04, data59)
