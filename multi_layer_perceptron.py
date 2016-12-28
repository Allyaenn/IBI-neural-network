import gzip # pour décompresser les données
import pickle
import numpy # pour pouvoir utiliser des matrices
import pylab # pour l'affichage

#paramètrage
nbStep = 2; #nombre d'itérations
nbLayer = 5; #nombre de couches dans le réseau de neurones
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
current_tmp = numpy.empty((0,0))
prior_tmp = numpy.empty((0,0))

for i in range (0,nbStep) : # sur n pas de temps
    for j in range (0,nbLayer) : #pour chaque couche du réseau application de la fonction d'activation
            #TODO : structuration du calcul en fonction du nombre de couche
            if j == 0 : #si on est sur la première couche, on prend l'entrée vec
                print ("IF 1 - Premier cas")
                print(vec.shape)
                print (weight_matrix_list[j].shape)
                current_tmp = numpy.dot(weight_matrix_list[j], vec) #le produit matriciel remplace la somme
                current_tmp = 1/(1 + numpy.exp(-current_tmp))
                print (current_tmp.shape)
            else : #sinon, on prend les données calculées dans la couche précedente
                print ("IF 1 - Deuxieme cas")
                print ("WML : ")
                print (weight_matrix_list[j].shape)
                print ("PRIOR")
                print (prior_tmp.shape)
                current_tmp = numpy.dot(weight_matrix_list[j], prior_tmp) #le produit matriciel remplace la somme
                current_tmp = 1/(1 + numpy.exp(-current_tmp))
                print ("CURRENT")
                print (current_tmp.shape)

            if j == nbLayer-1 : #si on est sur la dernière couche, on est sur la sortie -> pas d'ajout de biais, mise dans out
                print ("IF 2 - Premier cas")
                out = current_tmp
                print(out)
            else : #attention au biais -> ajouter 1 à la fin du tableau si ce n'est pas la dernière couche
                print ("IF 2 - Deuxieme cas")
                prior_tmp = numpy.append(current_tmp,1)
                print ("CURRENT")
                print (current_tmp)
                print ("PRIOR")
                print (prior_tmp)

    #calcul de l'erreur pour la couche de sortie
    #tab = numpy.empty(out.shape)
    #print(out.size)
    #print(tab.size)
    #for k in range (0,out.size) :
        #tab[k] = out[k]*(1-out[k])*(data[0][1][0][k]-out[k])

    #print (tab)

    #retropropagation des poids
