# Portage Rust/Linux de spectroid

## Bibliothèques

* wgpu
* winit
* realfft
* jack
* _un ring buffer concurrent ?_


## Algorithme

### Thread console

* Attend la condition variable qui signale les erreurs.
* Lit les différents compteurs d'erreurs.
* Compare aux derniers snapshots de compteurs.
* Signale les nouvelles erreurs par logs console.
* Met à jour les snapshots de compteurs.

### Thread audio

* Attend les paquets de audio entrants.
* Les écrit au bout d'un ring buffer concurrent "assez grand" qui va vers
  le thread FFT.
* En cas d'overrun (le thread d'en face n'a pas été rapide), écrase d'anciennes
  données mais signale l'erreur par incrément d'un compteur + ping condition
  variable du thread console.
    * Cette erreur ne devrait pas arriver avec un buffer bien dimensionné
      puisque le thread FFT s'exécute tout le temps. C'est donc ok si une
      corruption momentanée des donnée survient dans ce cas.
* Retourne attendre les paquets audio.

### Thread DSP

* S'exécute régulièrement selon un timer au rythme spécifié par l'utilisateur
  (ça aurait du sens que ce soit un multiple de la fréquence de rafraîchissement
  écran, par défaut 1x, + => 2x, 3x, 4x, - => 1/2x, 1/3x, 1/4x...).
* Copie les nouveaux samples du thread audio dans son ring buffer interne, de la
  taille de la plus grand FFT à effectuer, et les marque comme lus pour le
  thread audio.
* "Déplie" son ring buffer interne en un buffer linéaire pour la FFT.
* _TODO: Ajouter la décimation => Plusieurs tampons FFT à gérer basés sur les
  N/2, N/4... derniers samples._.
* Applique la fonction fenêtre choisie au tampon _(dans la v1, fenêtre rectangulaire)_
* Calculer la FFT.
* Ajoute les amplitudes de la FFT pour 20Hz+ à une ligne de "ring matrix"
  concurrente partagée avec le thread graphique, en les normalisant +
  convertissant en dBm:
    * Multiplier les bins FFT par 1/N _+ renormalisation liée à la fenêtre aussi_.
    * Calculer leur norm_sqr.
    * Calculer le log de tout ça.
    * Multiplier les logs résultants par 10 (on a déjà 2x via la norm_sqr).
* _TODO: Ajouter algos de réduction de bruit FFT basés sur la conservation d'un
  tampon circulaire d'amplitudes FFTs précédentes ici + regarder la tronche du
  bruit et en fonction essayer median filtering ou autre_.
* En cas d'overrun (le thread graphique ne rafraîchit pas assez vite), écrase
  d'anciennes données en râlant un coup comme ci-dessous. Mais fournit également
  au thread graphique un moyen de savoir que ça s'est produit, par exemple en
  utilisant un indice de ring buffer 64-bit non modulaire.
    * Le taux de rafraîchissement du thread graphique peut varier dans le temps,
      par exemple certaines APIs interrompent le rafraîchissement pendant le
      redimensionnement des fenêtres. Il est donc malheureusement envisageable
      que des données soient perdues de cette façon, même avec un buffer très
      grand (plusieurs secondes par exemple).
    * Dans ce cas, il faut que le thread graphique assume son échec en affichant
      clairement une bande de données manquantes dans sa visualisation, à la
      manière des bandes noires de spectroid, plutôt qu'une sortie corrompue...
* Retourne attendre le tick de timer suivant.
* 

### Thread graphique/événements

* S'exécute à chaque cycle de rafraîchissement de l'API graphique utilisée
  (probablement via wgpu + winit). Plus généralement, fait gestionnaire d'evts.
* Récupère les données du thread DSP dans sa "ring matrix" interne, de la taille
  de la longueur de l'historique affiché à l'écran en pixels, et les marque
  comme lues pour le thread DSP.
    * Si des données ont été perdues par buffer overrun, marquer les lignes
      correspondantes de la ring matrix à zéro.
* Copier l'ensemble de la ring matrix sur le GPU _(pour la v1, ensuite on
  essaiera d'être plus futé et de ne copier que ce qui a changé)_ comme texture
  avec interpolation linéaire, en "linéarisant" au passage.
* Afficher un quad avec un fragment shader qui...
    * Accède à la texture des amplitudes FFT en échelle log.
    * Traduit les amplitudes en couleurs via une fonction palette.
    * _Après la v1, essayer de ne rafraîchir que ce qui a changé._
* Par-dessus, faire le plot du dernier spectre en date avec des lignes
  antialiasées (cf exemples wgpu pour une idée).
* Commencer par hardcoder une échelle verticale raisonnable : 0dBm au max
  à `-20*bit_depth*log(2)` au min.
* _En-dehors des événements graphiques, gèrera à terme les événements associés
  aux interactions utilisateur, par exemple on pourrait imaginer que quand on
  appuie sur "+", ça augmente la résolution temporelle en augmentant la
  fréquence du thread DSP, et quand on appuie sur "-", ça la diminue._
    * _Certaines interactions, entre autres les redimensionnements de fenêtres,
      vont nécessiter des réallocations. Il va falloir de l'algorithmique
      lock-free assez chiadée pour gérer ça sans interrompre l'acquisition ni
      introduire des locks dans les threads temps réel... donc ce sera pas pour
      la v1. Et ce d'autant plus qu'il faudra offloader les trucs qui ne sont
      pas RT-safe, comme les allocations mémoire, à un autre thread avec de la
      synchro lock-free, ce qui va pas non plus être fun..._
    * _Une interaction "facile" et assez sympa serait de permettre le réglage
      de l'échelle verticale par cliquer-glisser sur le spectre live, le
      changement d'échelle s'appliquant aussi au spectrogramme, avec un reset
      des deux échelles possible via clic droit au même endroit_.
* Retourne attendre des événements.
* 


## Planif développement

* [X] Milestone 1: Thread audio jack qui est réveillé par jackd, remplit un ring
  buffer et signale les overrun qui vont inévitablement arriver avec un
  thread console.
    * Validation: Les overrun doivent être signalés.
* [X] Milestone 2: Thread DSP (RT!) qui se réveille à une fréquence de
  rafraîchissement hardcodée en const, réceptionne les données audio, maintient
  un ring buffer interne de taille N = 2^k hardcodée en const, calcule une
  realfft de radix N sans fenêtrage, et publie l'amplitude dBm de la dernière
  FFT en date dans un triple buffer. Gestion d'erreurs qui détecte quand la
  fréquence du thread DSP est supérieure à celle du thread audio, affiche un
  message d'erreur et refuse de démarrer le programme.
    * Validation: Ajout d'un thread console qui affiche la FFT toutes les
      secondes + soumission de divers signaux test via amsynth.
* Milestone 3: Thread graphique (RT!) qui fait un plot de cette dernière FFT en
  date à chaque événement de refresh écran via des lignes antialiasées en jaune
  sur fond noir. Dans un premier temps, on affiche tous les bins en échelle
  linéaire, sans légende, avec une échelle verticale hardcodée en const, et on
  prend toute la fenêtre. On devrait pouvoir s'inspirer très directement de
  l'exemple de lignes antialiasées de wgpu. Modifier la fréquence du thread DSP
  pour être un multiple ou un sous-multiple de la fréquence de rafraîchissement
  graphique.
    * Validation: Comme avant, mais avec la visu graphique plutôt qu'un thread
      console supplémentaire.
* Milestone 4: Passage à une échelle horizontale log contrôlé par une constante
  (à la base hardcodée via une const), puis ajout de légendes horizontales et
  verticales et d'une grille de pas en pixel raisonnable. Restriction des bins
  affichés à 20Hz-20kHz.
    * Validation: Comme avant, mais en plus c'est facile de vérifier la
      calibration.
* Milestone 5: Remplacement du triple buffer à l'interface DSP->graphique par un
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
* Milestone 6: Intégration de la réduction de bruit au thread DSP, qui conserve
  maintenant lui-même un historique des N (hardcodé en const) dernières FFTs
  qu'il a calculé et envoie des amplitudes filtrées au thread graphique.
    * Validation: Visualisation des FFTs filtrées, ajustement de la taille de
      fenêtres et test de différents algos de filtrages si besoin.
* Milestone 7: Ajout d'un plot spectrogramme au thread graphique. D'abord en
  échelle horizontale linéaire, avec une palette noir -> blanc, et avec un
  rafraîchissement intégral à chaque cycle. Approche à base de quad rendu par un
  fragment shader selon texture correspondant à l'historique d'amplitude FFT.
    * Validation: Test sur signaux audio synthétiques d'amsynth.
* Milestone 8: Meilleure palette de couleur (utiliser palette `INFERNO` tirée
  de matplotlib via la crate `palette`), rafraîchissement restreint aux
  nouvelles données, et support de l'échelle log dans le spectrogramme via un
  fragment shader plus intelligent.
    * Validation: Test sur signaux audio synthétiques d'amsynth.
* Milestone 9: Ajout de fenêtrage et normalisation associée dans le thread DSP,
  faire un peu de biblio sur wikipedia pour savoir quelles fenêtres tester.
    * Validation: Test sur signaux audio synthétiques, puis réels.
* Milestone 10: Ajout de décimation dans le thread DSP, compositing des
  différentes décimations en une amplitude unique soumise au thread graphique,
  du point de vue de qui rien ne devrait changer.
    * Validation: Test sur signaux audio synthétiques.
* Milestone 11: Essayer d'aller souplement d'une décimation à la suivante en
  faisant une moyenne pondérée (ex: supposons que le bin de fréquence plus
  élevée soit sur une fenêtre N=64 et celui d'après soit sur une fenêtre N=128,
  alors du dernier point absolu M au dernier point de la première fenêtre M-64
  on fait une moyenne pondérée entre les deux fenêtres selon à quel point on est
  proche de k=M-64).
    * Validation: Enregistrer des signaux réels transitoires (ex: chocs),
      vérifier qu'il n'y a plus de discontinuités marquées dans le spectrogramme.
* Milestone 12: Supporter le redimensionnement de la fenêtre sans perdre le
  caractère RT du thread graphique: les nouvelles allocations de mémoire vidéo
  sont faires par un thread séparé et remplacées à la volée par un algo qui est
  lock-free pour le thread graphique. Attention à la libération mémoire qui
  doit attendre que le thread graphique ait fini (mais je crois que wgpu le
  gère en standard).
    * Validation: Test sur signaux audio synthétiques, redim. la fenêtre.
* Milestone 13: Progressivement rendre des paramètres const ajustables à la
  volée par des raccourcis clavier, sans perdre la RT-safety (ex: les
  allocations mémoire sont faites par un thread séparé et remplacées à la volée
  par un algo qui est lock-free pour les threads RT). Au fur et à mesure, créer
  et mettre à jour une "ligne de statut" au-dessus du spectre instantané qui
  indique les paramètres actuels.
    * Validation: Test sur signaux audio synthétiques en appuyant sur des
      raccourcis clavier.
* Milestone 14: Ajouter un détecteur de pics au thread DSP. Soit la réduction de
  bruit utilisée rend ça facile, soit il faudra potentiellement appliquer une
  réduction de bruit plus agressive en interne pour la détection des pics, ou
  découpler la réduction de bruit du spectre instantané de celle du
  spectrogramme pour éviter de rendre le spectrogramme trop flou au niveau
  temporel.
    * Validation: Vérifier que les pics sont bien détectés de façon fiable.
* Milestone 15: Permettre un affichage en phones, moyennant une calibration
  comprenant une mesure de sweep à la position d'écoute pour avoir la courbe EQ
  de la pièce en régime établi + une calibration basée sur la mesure du volume
  sonore d'un signal 0dB/1kHz par un dBmètre. Utiliser les courbes isosoniques
  ISO 226:2003.
    * Validation: Confirmer que lors d'un sweep en fréquence de signal
      synthétique également envoyé sur les enceintes, la variation de
      l'intensité de couleur correspond à la variation de niveau sonore perçu.
* Milestone ???: Ping Carl Reinke, l'auteur de spectroid, sur LinkedIn pour lui
  faire part de ma réalisation.
* 
