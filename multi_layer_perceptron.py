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

vec = data[0][0][0]
weight = numpy.zeros((vec.size + 1,10))
vec = numpy.append(vec, 1)

print(weight.size)
#on commence avec une seule couche de 10 neurones pour commencer
out = [10]

for i in range (0,nbStep) : # sur n pas de temps
    for j in range (0,nbLayer) : #pour chaque couche du réseau
        #for k in range (0,10) : #pour chaque neurone de la couche
            #calcul de la sigmoide
            calc = numpy.dot(numpy.transpose(weight), vec) #le produit matriciel remplace la somme
            calc = 1/(1 + numpy.exp(-calc))
            print(calc)


#pour n pas de temps

#propagation

#calcul de l'erreur et back propagation


#C'est quoi les poids, on commence à 0 ?
