import gzip # pour décompresser les données
import pickle
import numpy # pour pouvoir utiliser des matrices
import pylab # pour l'affichage
import copy

#paramètrage
nbStep = 10; #nombre d'itérations
nbLayer = 3; #nombre de couches dans le réseau de neurones
nbNeurons = 10; #nombre de neurones par couche
learningRate = 0.1; #taux d'apprentissage

#chargement de la base de données.
f = gzip.open('mnist.pkl.gz')
data = pickle.load(f,encoding='latin1')

#comment naviguer dans la base de données ?
# 1ère colonne : test ou apprentissage
# 2ème colonne : image ou label
# 3ème colonne : numéro de l'exemple d'apprentissage

#selection d'une entrée au hasard dans la base

vec = data[0][0][0]
vec = numpy.append(vec, 1)

#intialisation de la structure de données
weight_matrix_list = []
for i in range (0, nbLayer) :
    if i == 0 :
        weight_matrix_list.append(numpy.zeros((nbNeurons,vec.size)))
    else :
        weight_matrix_list.append(numpy.zeros((nbNeurons,nbNeurons + 1))) #1 ligne contient les poids qui permette de construire le neurone d'arrivée
#TODO : initialiser les poids de façon random

#déclaration des structures qui vont stocker les valeurs des neurones
out = numpy.empty((0,0))
tmp = numpy.empty((0,0))

for i in range (0,nbStep) : # sur n pas de temps
    neuronsVL = []
    #forward
    for j in range (0,nbLayer) : #pour chaque couche du réseau application de la fonction d'activation
            #TODO : structuration du calcul en fonction du nombre de couche
            if j == 0 : #si on est sur la première couche, on prend l'entrée vec
                tmp = numpy.dot(weight_matrix_list[j], vec) #le produit matriciel remplace la somme
            else : #sinon, on prend les données calculées dans la couche précedente
                tmp = numpy.dot(weight_matrix_list[j], neuronsVL[j-1]) #le produit matriciel remplace la somme

            tmp = (1/(1 + numpy.exp(-tmp)))

            if j == nbLayer-1 : #si on est sur la dernière couche, on est sur la sortie -> pas d'ajout de biais, mise dans out
                out = tmp
            else : #attention au biais -> ajouter 1 à la fin du tableau si ce n'est pas la dernière couche
                neuronsVL.append(numpy.append(tmp,1))
    print (out)
    #backward
    for l in reversed (range(0,nbLayer)) :
        if l == nbLayer-1 :
            #calcul de l'erreur pour la couche de sortie
            err = numpy.empty(out.shape)
            for k in range (0,out.size) :
                err[k] = out[k]*(1-out[k])*(data[0][1][0][k]-out[k])
            #print (err)
            #correction des poids liés à la couche de sortie
            weight_old = copy.deepcopy(weight_matrix_list[l])
            err_old = copy.deepcopy(err)
            delta = numpy.dot(numpy.reshape(err,(err.size,1)), numpy.transpose(numpy.reshape(neuronsVL[l-1],(neuronsVL[l-1].size,1))))
            weight_matrix_list[l] = weight_matrix_list[l] + (delta*learningRate)

        else :
            #print (l)
            #retropropagation de l'erreur
            for k in range (0,neuronsVL[l].size-1) :
                #print(numpy.transpose(numpy.reshape(err_old,(err_old.size,1))).shape)
                #print(numpy.reshape(weight_old[k][:-1],(weight_old[k][:-1].size,1)).shape)
                som = numpy.dot(numpy.transpose(numpy.reshape(err_old,(err_old.size,1))),numpy.reshape(weight_old[k][:-1],(weight_old[k][:-1].size,1)))
                err[k] = neuronsVL[l][k] * (1 - neuronsVL[l][k]) * som
            #print (err)
            #correction poids
            if l != 0 :
                weight_old = copy.deepcopy(weight_matrix_list[l])
                err_old = copy.deepcopy(err)
                delta = numpy.dot(numpy.reshape(err,(err.size,1)), numpy.transpose(numpy.reshape(neuronsVL[l-1],(neuronsVL[l-1].size,1))))
                weight_matrix_list[l] = weight_matrix_list[l] + (delta*learningRate)
                #print (weight_matrix_list[l])
            else : # l == 0
                delta = numpy.dot(numpy.reshape(err,(err.size,1)), numpy.transpose(numpy.reshape(vec,(vec.size,1))))
                weight_matrix_list[l] = weight_matrix_list[l] + (delta*learningRate)
                #print (weight_matrix_list[l])
