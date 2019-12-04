# Report

## General notes: problems encountered and resolution/interpretation.
Le principal était que quand on runnait notre fichier project.py sur la fonction constante (qui contenait l'entraînement du réseau de neurones et les tests), on n'obtenait pas toujours le même résultat.
Ce premier point est réglé : nous avons séparé en deux fichiers distincts l'entraînement et le test. Dans le fichier training, on a modifié l'entraînement en faisait en sorte que le réseau s'entraîne sur plusieurs époques, une dizaine (il ne s'entraînait auparavant que sur une époque). On a ajouté une ligne permettant d'enregistrer l'entraînement du modèle après chaque époque ce qui nous permet également de voir l'évolution de la fonction loss après chaque époque (pour vérifier qu'il apprend bien).
Dans le fichier testing, on charge le modèle qu'on a entraîné et sauvegardé dans le fichier training et on plot les résultats.

## Tests: graphs and analysis
### Tests on `data_constant_2000.pkl`
