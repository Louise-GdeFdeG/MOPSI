1. Fonction polynomiale/sinusoïde : cinfini

  On voit sur l'exemple de l'approximation de la sinusoïde que le modèle est performant. L'approximation est très proche de la fonction d'origine. [fichier `testing_sinus_10_5_2000_40_0.0001.png`]

  Sur l'exemple polynomial, on voit l'effet du nombre de couches sur la précision dans le meilleur cas.
  Pour W, Nepoch, lr constant l'approximation dans le meilleur cas est bien meilleure pour L=7 que pour L=2.
  [fichiers `testing_polynomial_5_2_2000_20_0.0001.png` et `testing_5_7_2000_20_0.001ter.png`]

  Pourtant, lorqu'on trace l'erreur en fonction de L (fichier eL_polynomial), on voit qu'à partir de L=5, l'erreur augmente à nouveau.
  Nous l'expliquons avec deux arguments :
  - Le phénomène de saturation (expliqué dans les remarques générales) apparaît plus fréquemment lorsqu'on augmente les dimensions du modèle.
  - On est sur un nombre d'époques faible, peut-être trop faible pour que les réseaux à grande dimensions aient le temps de bien se calibrer.
  (fichier `eL_polynomial.png`)

2. Fonction chapeau sans trou : continue, dérivée discontinue
Bonne approximation avec W faible, L faible, et nepoch assez faible. (W=5, L=4, Nepoch = 10, lrate = 0.001).
[fichier `testing_hat_5_4_2000_10_0.001.png`]

3. Fonction chapeau avec trou : pas continue
Dans cette partie, on garde la même fonction objectif, ie la fonction chapeau.
Par contre, on enlève du training set tous les points autour de la pointe du chapeau, sur un segment de largeur gap_width. Le but est de voir si le réseau peut extrapoler un point de discontinuité à partir de données qui l'entourent.
Le modèle se comporte plutôt bien dans ce genre de cas.
Pour gap_width petit (0.1 ; 0.2), le réseau approxime la pointe du chapeau par un palier constant. [fichier ` testing_truncaded_hat_W5_L4_Nepoch10_lr0.001_gap_width02.png` et `testing_truncated_hat_W5_L4_Nepoch10_lr0.001_gap_width01.png`]

Lorsqu'on augmente la taille de gap_width, le comportement est un peu plus aléatoire. Mais le cas le plus fréquent est toujours un plateau constant sur le segment où il n'y a pas de données d'entraînement. Quelquefois, le réseau fitte avec la fonction objectif, mais cela semble plus être un effet du hasard qu'un résultat vraiment exploitable.
[fichiers `testing_hat_W5_L4_Nepoch10_lr0.001_gap_width04.png` et `testing_truncated_hat_W5_L4_Nepoch10_lr0.001_gap_width04.png` et `testing_truncated_hat_W5_L4_Nepoch10_lr0.001_gap_width07.png` et `testing_truncated_hat_W50_L7_Nepoch=10_lr=0.001_gap_width07.png` ]


4. Fonction escalier avec un point de discontinuité : non continue, dérivée hyper discontinue (piecewise2)

Nos modèles ont été très performants dans ce cas.
En effet, avec des valeurs faibles de W et L, on arrive déjà à une approximation correcte.
[`testing_piecewise2_W7_L4_2000_10.png`]

5. Fonction escalier avec plusieurs points de discontinuité : non continue, dérivée hyper discontinue (piecewise3,4,10)
Pour les fonctions avec peu de points de discontinuité (piecewise3 et piecewise4), on arrrive à une approximation satisfaisante, pour des valeurs de WxL plus grandes, et un learning_rate beaucoup grand (0.001).
[`testing_piecewise3_W100_L4_2000_10_0001.png`]
Mais pour la fonction piecewise qui a 9 points de discontinuité, aucun paramètre ne nous a permis d'obtenir une approximation qui imite la discontinuité. Visiblement, cette fonction ne peut pas être gérée par notre réseau. Nous en faisons une interprétation dans la partie "remarques générales".
[` piecewise10_W100_L10_2000_10_lr0001.png`]


7. Remarques générales

On sait qu'en théorie, la précision est une fonction croissante de W et L.
En pratique, ce n'est pas toujours ce que nous avons observé.
Une explication possible est que nous ne disposions pas de puissance de calcul suffisante pour faire tourner nos calculs suffisamment longtemps.
En effet, lorsque W et L augmentent, il y a plus de paramètres à régler. Il faut donc plus d'époques au réseau pour qu'il s'ajuste.
Ne disposant pas de serveurs de calculs, nous avons dû nous limiter à un nombre d'époques de l'ordre de la dizaine. Ce n'était pas suffisant pour obtenir une précision satisfaisante dès que nos réseaux grandissaient.

D'autre part, les réseaux neuronaux ont la réputation de mieux gérer les discontinuités que les méthodes traditionnelles (régression, interpolation).
Nous l'avons vérifié dans 2 cas : unne fonction continue de dérivée discontinue (chapeau), et une fonction discontinue en peu de points(fonction escalier simple fonctions escaliers à 2 et 3 points de discontinuité).
Pour ces fonctions, on arrive à des approximations satisfaisantes.

Mais dès qu'il y a trop de discontinuité, nos réseaux ont du mal à suivre. Cela est flagrant pour la fonction escalier avec 9 points de discontinuités. Quand il y a trop de discontinuité, c'est comme si le réseau paniquait et se contentait d'une régression linéaire.
Nous pensons que c'est à cause des limites intrinsèques de nos modèles. Nos modèles sont des combinaisons de fonctions linéaires et de relus.
Ces fonctions génèrent fatalement des fonctions continues. Mais elles peuvent facilement créer des fonctions dont la dérivée est discontinue, avec une discontinuité finie. Lorsque la fonction objectif présente peu de points de discontinuités, nos modèles suffisent. Mais passé un certain seuil, nos modèles ne sont plus suffisants.

Dans un premier temps, nous n'avons pas étudié l'influence du learning rate sur l'efficacité de nos modèles. Nous avons surtout fait varier W, L, et le nombre d'époques.
Cela nous a permis d'atteindre des résultats plutôt satisfaisants sur les fonctionc Cinfini.
Ce que nous avons pu voir, c'est que plus le learning rate est bas, plus le modèle est capable de précision. Mais plus le learning rate est bas, plus il faut de calculs pour atteindre la précision optimale.
Pour approximer des fonctions Cinfini, il convenait de prendre un learning_rate faible(de l'ordre de 10^-4), pour suivre la courbe avec précision. Mais pour approximer des fonctions avec de la discontinuité, il valait mieux prendre un learning_rate plus grand, sans quoi les approximations peinaient a avoir des courbes de pente suffisament grande.

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
