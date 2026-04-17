
## Setup

posizionati nella cartella root del progetto
`cd coppa_america_rl`

Crea l'ambiente virtuale isolato eseguendo:
`python -m venv venv`

Attiva l'ambiente virtuale appena creato. Se usi Windows (Prompt dei comandi o PowerShell), esegui:
`venv\Scripts\activate`
Se invece usi macOS o Linux, esegui:
`source venv/bin/activate`

Una volta che l'ambiente è attivo, installa tutte le dipendenze bloccate nel file di testo:
`pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`

Nota: potrebbero mancare dei pacchetti, se non funziona scrivetemi

## avvio
Attiva l'ambiente
`venv\Scripts\activate`

Avvia lo script
` python main.py `

## Struttura dei File


**config.py**
Contiene tutte le costanti: iperparametri del modello PPO, limiti del campo, intensità del vento, parametri di rendering. Tutte le modifiche ai valori del simulatore vanno fatte obbligatoriamente qui.

**main.py**
Non contiene logica, ma serve solo ad avviare le due macro-fasi del progetto (Addestramento o Generazione Video).
Se ti serve solo il training o la generazione del video puoi commentare una delle due righe di codice

**environment.py**
Contiene esclusivamente l'ambiente di simulazione `ImprovedSailingEnv` ereditato da `gym.Env`. Gestisce lo stato, i calcoli dei reward, la fisica di base e gli step temporali. Importa i parametri dinamici e le costanti direttamente dal file di configurazione.

**train.py**
Gestisce il setup del modello di Reinforcement Learning (PPO) e la classe `SuccessTrackingCallback` necessaria per il monitoraggio a terminale delle performance e delle distanze durante il ciclo di apprendimento.

**video_generator.py**
Contiene la logica per caricare un modello addestrato (in formato zip) e fargli giocare episodi completi in fase di inferenza. Si occupa poi di salvare i frame generati tramite `imageio` in un file video finale.

**utils.py**
Libreria di funzioni matematiche, tra cui il calcolo delle intersezioni, la normalizzazione degli angoli e il calcolo delle velocità polari della barca.

**render_utils.py**
Isola interamente la logica di disegno grafico gestita da `matplotlib`, occupandosi di disegnare il campo, la barca, la scia e i testi informativi a schermo. Viene richiamato dall'ambiente di simulazione durante la chiamata alla funzione di render.

**requirements.txt e .gitignore**
File di servizio per il versioning tramite Git e la riproducibilità su altre macchine. L'ambiente `venv`, i file di log di TensorBoard e i file multimediali pesanti vengono ignorati per mantenere la repository pulita.