# objectDetection_YOLOv8

## requirements.txt
Inštalácia balikov

    pip install -r requirements.txt

## download_dataset.py
Súbor sa používa na získanie obrázkov z platformy Roboflow.

## main.py
Trénovanie modelu YOLOv8.

## test_model.py
Vstupom pre súbor je natrénovaný model a testovací obrázok, 
na ktorom sa otestuje natrénovaný model.

Výstupom súboru test_model.py je obrázok, v ktorom sú objekty rozpoznané vstupným
modelom použitím knižnice opencv-python, ktorý vytvorí ohraničujúce boxy pre
objekt. A tiež opis triedy, do ktorej obrázok patrí.

Na prácu s týmto súborom potrebujete knižnicu opencv-python s verziou - 4.5.5.62

Je potrebné použiť príkaz

    pip install opencv-python==4.5.5.62

## cm_test_data.py

Vstupom je testovací priečinok obsahujúci 3 podpriečinky. Každý z týchto podpriečinkov
obsahuje názov triedy (not_hold, one_hand_hold a two_hand_hold). Každý z týchto podpriečinkov
musí obsahovať testovacie fotografie. Posledným vstupným parametrom je natrénovaný model.

Výstupom je konfúzna matica tried (not_hold, one_hand_hold a two_hand_hold).