1. Fonction polynomiale/sinusoïde : cinfini
  Plot de eL : pas très concluant.
  L'approximation est optimale pour L entre 2 et 4.
  Une explication possible est que si L est trop grand, il y a un trop grand nombre de paramètres à régler.

  On voit sur l'exemple de l'approximation de la sinusoïde que le modèle est performant. L'approximation est très proche de la fonction d'origine.

  Sur l'exemple polynomial, on voit l'effet du nombre de couches sur la précision.
  Pour W, Nepoch, lr constant l'approximation dans le meilleur cas est bien meilleure pour L=7 que pour L=2.

  Pourtant, lorqu'on trace l'erreur en fonction de L (fichier eL_polynomial), on voit qu'à partir de L=5, l'erreur augmente à nouveau.
  Nous l'expliquons avec deux arguments :
  - Le phénomène de saturation expliqué dans les remarques générales apparaît plus fréquemment lorsqu'on augmente les dimensions du modèle.
  - On est sur un nombre d'époques faible, peut-être trop faible pour que les réseaux à grande dimensions aient le temps de bien se calibrer.

2. Fonction chapeau sans trou : continue, dérivée discontinue
Bonne approximation avec W faible, L faible, et nepoch assez faible.

3. Fonction chapeau avec trou : pas continue

4. Fonction escalier avec un point de discontinuité : non continue, dérivée hyper discontinue

Nos modèles ont été très performants dans ce cas.
En effet, avec des valeurs faibles de W et L, on arrive déjà à une approximation correcte.

5. Fonction escalier avec plusieurs points de discontinuité : non continue, dérivée hyper discontinue

6. Fonction nn

7. Remarques générales

On sait qu'en théorie, la précision est une fonction croissante de W et L.
En pratique, ce n'est pas toujours ce que nous avons observé.
Une explication possible est que nous ne disposions pas de puissance de calcul suffisante pour faire tourner nos calculs suffisamment longtemps.
En effet, lorsque W et L augmentent, il y a plus de paramètres à régler. Il faut donc plus d'époques au réseau pour qu'il s'ajuste.
Ne disposant pas de serveurs de calculs, nous avons dû nous limiter à un nombre d'époques de l'ordre de la dizaine. Ce n'était pas suffisant pour obtenir une précision satisfaisante dès que nos réseaux grandissaient.

D'autre part, les réseaux neuronaux ont la réputation de mieux gérer les discontinuités que les méthodes traditionnelles (régression, interpolation).
Nous l'avons vérifié dans 2 cas : unne fonction continue de dérivée discontinue (chapeau), et une fonction discontinue en 1 point(fonction escalier simple).
Pour ces deux fonctions, on arrive à des approximations satisfaisantes pour W, L et nepoch faibles.
Mais dès qu'il y a trop de discontinuité, nos réseaux ont du mal à suivre. Cela est flagrant pour les fonctions escalier avec plusieurs points de discontinuité.
Nous pensons que c'est à cause des limites intrinsèques de nos modèles. Nos modèles sont des combinaisons de fonctions linéaires et de relus.
Ces fonctions génèrent fatalement des fonctions continues. Mais elles peuvent facilement créer des fonctions dont la dérivée est discontinue, avec une discontinuité finie.

Dans un premier temps, nous n'avons pas étudié l'influence du learning rate sur l'efficacité de nos modèles. Nous avons surtout fait varier W, L, et le nombre d'époques.
Cela nous a permis d'atteindre des résultats plutôt satisfaisants. Mais une étude plus approfondie du learning_rate nous permettrait d'avoir des résultats encore plus fins.
Ce que nous avons pu voir, c'est que plus le learning rate est bas, plus le modèle est capable de précision. Mais plus le learning rate est bas, plus il faut de calculs pour atteindre la précision optimale.
Nous avons encore une fois été limités par la puissance de calcul dont nous disposions. C'est pourquoi nous n'avons pas pu exploiter à fond l'analyse de l'effet du learning_rate.

Nous avons en outre fréquemment rencontré un phénomène étonnant, que nous avons baptisé "saturation".
Souvent, le réseau renvoie une fonction qui ne correspond pas du tout à la fonction objectif.
Dans ces cas, si on regarde l'évolution de l'erreur pendant la période d'entraînement, on remarque que l'erreur diminue fortement pendant les premières époques, puis qu'elle atteint rapidement un palier.
Une fois ce palier atteint, l'erreur ne redescend plus. Notre interprétation est la suivante :
Lorsque nous initialisons nos modèles, les paramètres de poids et de biais sont choisis aléatoirement entre -1 et 1. Il semble que quand les paramètres de départ sont trop mauvais, le réseau n'arrive pas à se corriger.
Nous pensons que c'est parce que lors de la mise à jour des poids, les valeurs d'erreur sont trop fortes, ce qui affecte le gradient, et empêche le système d'évoluer vers une mielleure solution.
Cela est visible sur le fichier "saturation"
Cela illustre le manque de fiabilité du machine learning dans certains cas.
Pour pallier à ce problème, nous pourrions obliger les modèles à s'initialiser avec des valeurs de paramètres fixées. Nous n'avons pas exploré cette piste, car elle allait à l'encontre de notre projet.
Notre projet était en effet d'étudier les capacités de réseaux neuronaux en leur laissant un maximum d'autonomie. Fixer les paramètres initiaux serait donc "tricher".
Mais dans une application professionnelle, il serait appréciable de diminuer au maximum les incertitudes des modèles. Cette méthode serait alors pertinente.
