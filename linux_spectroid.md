# Portage Rust/Linux de spectroid

## Algorithme

* _TODO: Ajouter algos de réduction de bruit FFT basés sur la conservation d'un
  tampon circulaire d'amplitudes FFTs précédentes ici + regarder la tronche du
  bruit et en fonction essayer median filtering ou autre_.

## Planif développement

* [ ] Passage à une échelle horizontale log contrôlé par une constante
  (à la base hardcodée via une const), puis ajout de légendes horizontales et
  verticales et d'une grille de pas en pixel raisonnable. Restriction des bins
  affichés à 20Hz-20kHz.
    * Validation: Comme avant, mais en plus c'est facile de vérifier la
      calibration.
* [ ] Remplacement du triple buffer à l'interface DSP->graphique par un
  ring buffer pouvant conserver un historique de FFTs. Signalement des
  overruns dans le thread console + compteur non modulaire 64-bit permettant au
  thread graphique de les détecter aussi. Récupération de l'historique dans le
  thread graphique, qui en conserve un autre de longueur plus importante, et
  marque les overruns par des amplitudes FFTs nulles (dB = -inf). Pour
  l'instant, la taille de ces deux historiques est hardcodée en const.
    * Validation: Envoi de signaux audio réels au programme et repérage de bin
      bruité, puis dump régulier de l'historique de ce bin dans la console.
      Analyse du bruit avec R pour voir ses caractéristiques temporelles et en
      histogramme: Blanc? Gaussien? Essai de différents algorithmes de filtrage:
      médiane glissante, moyenne glissante, autres filtres... Sélection de
      l'algorithme de filtrage adapté.
* [ ] Intégration de la réduction de bruit au thread DSP, qui conserve
  maintenant lui-même un historique des N (hardcodé en const) dernières FFTs
  qu'il a calculé et envoie des amplitudes filtrées au thread graphique.
    * Validation: Visualisation des FFTs filtrées, ajustement de la taille de
      fenêtres et test de différents algos de filtrages si besoin.
* [ ] Ajout d'un plot spectrogramme au thread graphique. D'abord en
  échelle horizontale linéaire, avec une palette noir -> blanc, et avec un
  rafraîchissement intégral à chaque cycle. Approche à base de quad rendu par un
  fragment shader selon texture correspondant à l'historique d'amplitude FFT.
    * Validation: Test sur signaux audio synthétiques d'amsynth.
* [ ] Progressivement rendre des paramètres const ajustables à la
  volée par des raccourcis clavier, sans perdre la RT-safety (ex: les
  allocations mémoire sont faites par un thread séparé et remplacées à la volée
  par un algo qui est lock-free pour les threads RT). Au fur et à mesure, créer
  et mettre à jour une "ligne de statut" au-dessus du spectre instantané qui
  indique les paramètres actuels.
    * Validation: Test sur signaux audio synthétiques en appuyant sur des
      raccourcis clavier.
* [ ] Ajouter un détecteur de pics au thread DSP. Soit la réduction de
  bruit utilisée rend ça facile, soit il faudra potentiellement appliquer une
  réduction de bruit plus agressive en interne pour la détection des pics, ou
  découpler la réduction de bruit du spectre instantané de celle du
  spectrogramme pour éviter de rendre le spectrogramme trop flou au niveau
  temporel.
    * Validation: Vérifier que les pics sont bien détectés de façon fiable.
* [ ] Permettre un affichage en phones, moyennant une calibration
  comprenant une mesure de sweep à la position d'écoute pour avoir la courbe EQ
  de la pièce en régime établi + une calibration basée sur la mesure du volume
  sonore d'un signal 0dB/1kHz par un dBmètre. Utiliser les courbes isosoniques
  ISO 226:2003.
    * Validation: Confirmer que lors d'un sweep en fréquence de signal
      synthétique également envoyé sur les enceintes, la variation de
      l'intensité de couleur correspond à la variation de niveau sonore perçu.
* [ ] Ping Carl Reinke, l'auteur de spectroid, sur LinkedIn pour lui
  faire part de ma réalisation.
