import gzip # pour décompresser les données
import pickle
import numpy # pour pouvoir utiliser des matrices
import pylab # pour l'affichage

#paramètrage
nbStep = 2;
nbLayer = 1;
learningRate = 0.5;

#chargement de la base de données.
f = gzip.open('mnist.pkl.gz')
data = pickle.load(f,encoding='latin1')

#comment naviguer dans la base de données ?
# (data[0][0][1])
# test ou apprentissage
# image ou label
#numéro de l'exemple d'apprentissage

#selection d'une entrée au hasard dans la base

#print (data[0][1][0][0])

vec = data[0][0][0]
vec = numpy.append(vec, 1)

#intialisation de la structure de données
weight_matrix_list = [nbLayer]
for i in range (0, nbLayer) :
    if i == 0 :
        weight_matrix_list[i] = numpy.zeros((vec.size,10))
    else :
        weight_matrix_list[i] = numpy.zeros((11,10))
#TODO : initialiser les poids de façon random

print ("Weight Matrix List : ")
print (len(weight_matrix_list))

#on commence avec une seule couche de 10 neurones pour commencer
out = numpy.empty((10,1))

for i in range (0,nbStep) : # sur n pas de temps
    for j in range (0,nbLayer) : #pour chaque couche du réseau
        #for k in range (0,10) : #pour chaque neurone de la couche
            #application de la fonction d'activation
            #TODO : structuration du calcul en fonction du nombre de couche
            out = numpy.dot(numpy.transpose(weight_matrix_list[j]), vec) #le produit matriciel remplace la somme
            out = 1/(1 + numpy.exp(-out))
            print(out)
            #attention au biais -> ajouter 1 à la fin du tableau si ce n'est pas la dernière couche

    #calcul de l'erreur pour la couche de sortie
    tab = numpy.empty(out.shape)
    print(out.size)
    print(tab.size)
    for k in range (0,out.size) :
        tab[k] = out[k]*(1-out[k])*(data[0][1][0][k]-out[k])

    print (tab)

    #retropropagation des poids


#pour n pas de temps

#propagation

#calcul de l'erreur et back propagation


#C'est quoi les poids, on commence à 0 ?
