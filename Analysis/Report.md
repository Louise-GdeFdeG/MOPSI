# Report

## General notes: problems encountered and resolution/interpretation.
Le principal était que quand on runnait notre fichier project.py sur la fonction constante (qui contenait l'entraînement du réseau de neurones et les tests), on n'obtenait pas toujours le même résultat.
Ce premier point est réglé : nous avons séparé en deux fichiers distincts l'entraînement et le test. Dans le fichier training, on a modifié l'entraînement en faisait en sorte que le réseau s'entraîne sur plusieurs époques, une dizaine (il ne s'entraînait auparavant que sur une époque). On a ajouté une ligne permettant d'enregistrer l'entraînement du modèle après chaque époque ce qui nous permet également de voir l'évolution de la fonction loss après chaque époque (pour vérifier qu'il apprend bien).
Dans le fichier testing, on charge le modèle qu'on a entraîné et sauvegardé dans le fichier training et on plot les résultats.

## Tests: graphs and analysis
### Tests on `data_constant_2000.pkl`
**How can we interpret the shape of the error function?**
* W = 5, L = 2, N = 2000, **NB_EPOCH = 1**, learning_rate = 0.0001  
![Comparison between the function and its approximation](/graphs/testing_constant_5_2_2000_1_0.0001.png)
![Error value](/graphs/error_constant_5_2_2000_1_0.0001.png)

* W = 5, L = 2, N = 2000, **NB_EPOCH = 5**, learning_rate = 0.0001  
![Sum of loss value](/graphs/training_validation_constant_5_2_2000_5_0.0001.png)
After 5 epochs, the sum of loss value of the training is roughly of 0.242.
![Comparison between the function and its approximation](/graphs/testing_constant_5_2_2000_5_0.0001.png)  
I don't know why the function has this shape... 
![Error value](/graphs/error_constant_5_2_2000_5_0.0001.png)


* W = 5, L = 2, N = 2000, **NB_EPOCH = 10**, learning_rate = 0.0001  
![Sum of loss value](/graphs/training_validation_constant_5_2_2000_10_0.0001.png)
After 10 epochs, the sum of loss value of the training is between 5.50e-3 and 5.25e-3.  
![Comparison between the function and its approximation](/graphs/testing_constant_5_2_2000_10_0.0001.png)
![Error value](/graphs/error_constant_5_2_2000_10_0.0001.png)

* W = 5, L = 2, N = 2000, **NB_EPOCH = 50**, learning_rate = 0.0001  
![Sum of loss value](/graphs/training_validation_constant_5_2_2000_50_0.0001.png)
After 50 epochs, the sum of loss value of the training is lower than 1e-5 and that of the validation is lower than 1e-8.
![Comparison between the function and its approximation](/graphs/testing_constant_5_2_2000_50_0.0001.png)
Gives the right value (they are all really close to 1), but does it give the right f(x) ?
![Error value](/graphs/error_constant_5_2_2000_50_0.0001.png)