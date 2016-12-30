import gzip # pour décompresser les données
import pickle
import numpy # pour pouvoir utiliser des matrices
import pylab # pour l'affichage
import copy
import random
import time

#pour rendre l'affichage plus lisible en supprimant l'affichage "scientifique"
numpy.set_printoptions(suppress=True)

#paramètrage
nbStep = 100000; #nombre d'itérations
nbLayer = 2; #nombre de couches dans le réseau de neurones (en comptant la couche de sortie)
nbNeurons = 10; #nombre de neurones par couche
learningRate = 0.1; #taux d'apprentissage

#chargement de la base de données.
print ("Chargement de la base de données...")
f = gzip.open('mnist.pkl.gz')
data = pickle.load(f,encoding='latin1')
print("Base de données chargée.")
#determination de la taille du vecteur d'entrée
inputSize = len(data[0][0][0]) + 1

#PHASE 1 : APPRENTISSAGE --------------------------------------------------------------------------

#intialisation de la structure de données
#TODO : initialiser les poids de façon random
weightML = []
for i in range (0, nbLayer) :
    if i == 0 :
        weightML.append(numpy.zeros((nbNeurons,inputSize)))
    else :
        weightML.append(numpy.zeros((nbNeurons,nbNeurons + 1)))

#déclaration des structures qui vont stocker les valeurs des neurones
out = numpy.empty((0,0))
tmp = numpy.empty((0,0))

print ("Début de l'apprentissage...")
t1=time.process_time()
for i in range (0,nbStep) : # sur n pas de temps
    #selection d'une entrée au hasard dans la base d'apprentissage
    randomInt = random.randint(0,len(data[0][0])-1)
    #print (randomInt)
    vec = data[0][0][randomInt]
    vec = numpy.append(vec, 1)
    neuronsVL = []
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
            for k in range (0,out.size) :
                err[k] = out[k]*(1-out[k])*(data[0][1][randomInt][k]-out[k])

            #correction des poids liés à la couche de sortie
            weight_old = copy.deepcopy(weightML[l])
            err_old = copy.deepcopy(err)
            delta = numpy.dot(numpy.reshape(err,(err.size,1)), numpy.transpose(numpy.reshape(neuronsVL[l-1],(neuronsVL[l-1].size,1))))
            weightML[l] = weightML[l] + (delta*learningRate)

        else :
            #retropropagation de l'erreur
            for k in range (0,neuronsVL[l].size-1) :
                som = numpy.dot(numpy.transpose(numpy.reshape(err_old,(err_old.size,1))),numpy.reshape(weight_old[k][:-1],(weight_old[k][:-1].size,1)))
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

t2=time.process_time()
print ("Apprentissage terminé en : " + str(t2 - t1) + " secondes")
#PHASE 2 : TEST --------------------------------------------------------------------------
print ("Début de la phase de test...")
bonnes_reponses = 0
reponses = 0

for i in range (0,len(data[1][0])) : #len(data[1][0])
    vec = data[1][0][i]
    vec = numpy.append(vec, 1)
    neuronsVL = []
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

    reponses = reponses + 1
    if numpy.argmax(out) == numpy.argmax(data[1][1][i]) :
        bonnes_reponses = bonnes_reponses + 1
    #print ("out : " + str(numpy.argmax(out)) + " data : " + str(numpy.argmax(data[1][1][i])))

print ("Taux de bonnes réponses : " + str(bonnes_reponses/reponses*100))
print ("Phase de test terminée.")
