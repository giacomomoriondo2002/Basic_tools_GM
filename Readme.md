# Basic tools GM

Repository contenente un piccolo pacchetto Python con funzioni di utilita per analisi numerica, processi stocastici, segnali 1D, filtraggio e gestione di dati tabulari.

## Dipendenze

Il modulo usa:

- `numpy`
- `scipy`

Il modulo `io.py` usa anche `datetime`, disponibile nella libreria standard di Python, per aggiungere data e ora nei file di testo generati.

## Come installare il modulo

Da bash, puoi installare il modulo direttamente dalla repository GitHub con:

```bash
pip install git+https://github.com/giacomomoriondo2002/Basic_tools_GM.git
```

e aggiornarlo, in caso di nuovi commit, con:

```bash
pip install --upgrade git+https://github.com/giacomomoriondo2002/Basic_tools_GM.git
```

## Struttura del pacchetto

Il pacchetto e diviso in sottomoduli tematici:

```text
basic_tools_GM/
├── stochastic.py
├── io.py
├── utils.py
├── convolution.py
├── linalg.py
├── peaks.py
├── signals.py
├── filtering.py
├── ode.py
└── module.py
```

Il file `module.py` e mantenuto come livello di compatibilita: reimporta le funzioni dai nuovi sottomoduli, quindi il vecchio stile di import continua a funzionare.

Esempi:

```python
from basic_tools_GM.filtering import optimize_filter
from basic_tools_GM.peaks import rel_peaks_finder
```

oppure, usando l'API aggregata:

```python
from basic_tools_GM import optimize_filter
from basic_tools_GM.module import optimize_filter
```

## Contenuto dei sottomoduli

### Processi stocastici

#### `autocovariance_matrix_1D_system(X, N_iter_axis=0, unbiased=True)`

Calcola la matrice di autocovarianza a due tempi di un processo stocastico 1D a partire da un insieme di traiettorie indipendenti.

- `X` contiene le realizzazioni del processo.
- `N_iter_axis` indica quale asse rappresenta le traiettorie indipendenti.
- `unbiased=True` usa il fattore di normalizzazione `N-1`; con `False` usa `N`.

Restituisce una matrice `(n_time, n_time)` in cui ogni elemento rappresenta la covarianza tra due istanti temporali.

#### `Random_jump(W, current_state, State_vec, test_W=True)`

Estrae casualmente il prossimo stato di un processo di Markov discreto usando una matrice di transizione `W`.

- `W[i, j]` e la probabilita di passare dallo stato `j` allo stato `i`.
- `current_state` e lo stato attuale.
- `State_vec` contiene tutti gli stati possibili, scalari o vettoriali.
- `test_W=True` verifica che ogni colonna di `W` sommi a 1.

Restituisce un dizionario con l'indice del nuovo stato e il valore dello stato estratto.

#### `autocovariance_function(x, max_lag, max_lag_mode='discrete', dt=None)`

Calcola la funzione di autocovarianza temporale di una singola serie 1D usando una correlazione basata su FFT.

- `max_lag_mode='discrete'` interpreta `max_lag` come numero di campioni.
- `max_lag_mode='time'` interpreta `max_lag` come tempo fisico e richiede `dt`.
- Il risultato non e normalizzato: per ottenere una autocorrelazione normalizzata si puo dividere per il valore a lag zero.

### Input/output e utility di base

#### `txt_file_writer(filename, names_of_arrays, *arrays, digit_of_precision='all', mode='f', date=True, log_file=None, delimiter='\t', auto_ordering=True)`

Scrive piu array come colonne allineate in un file di testo.

- gestisce array di lunghezze diverse aggiungendo valori mancanti;
- scrive una riga di intestazione con i nomi delle colonne;
- puo includere data, ora e una descrizione dei dati;
- permette di scegliere precisione numerica, formato e delimitatore;
- puo ordinare automaticamente le colonne per lunghezza decrescente.

Restituisce `True` se la scrittura va a buon fine.

#### `couple_sorting(x, y, descending=False)`

Ordina due sequenze in parallelo usando `x` come riferimento per l'ordinamento.

- `x` viene ordinato.
- `y` viene riordinato seguendo gli stessi indici di `x`.
- `descending=True` ordina in ordine decrescente.

Restituisce la coppia `(x_sorted, y_sorted)`.

#### `best_index_finder(vec, *args, toll='auto', return_vec='best', mode='strict', ignore_length_warning=False)`

Trova gli indici di `vec` piu vicini a uno o piu valori target.

- `return_vec='best'` restituisce l'indice del punto piu vicino.
- `return_vec='all'` restituisce tutti gli indici entro la tolleranza.
- `toll='auto'` usa la distanza media tra punti consecutivi di `vec`.
- `mode='strict'`, `mode='only close points'` e `mode='loose'` controllano quanto rigidamente applicare la tolleranza.

E utile per associare valori continui a una griglia discreta di punti.

### Convoluzione, deconvoluzione e algebra lineare

#### `convolution(a, b, mode='full', veca=None, vecb=None)`

Calcola la convoluzione tra due segnali usando `scipy.signal.fftconvolve`.

- Supporta le modalita standard `full`, `same` e `valid`.
- Con `mode='scaled'` o `mode='full scaled'` usa i vettori temporali per scalare il risultato con il passo di campionamento.
- Con `mode='full scaled'` restituisce anche il vettore temporale della convoluzione.

#### `reg_deconvolution(trace, Green, tvec, *args, mode='discrete', epsilon=None)`

Esegue una deconvoluzione regolarizzata nel dominio delle frequenze.

- `Green` e una funzione di risposta valutata sui ritardi temporali.
- `epsilon` controlla la regolarizzazione numerica.
- `mode='continuous'` applica la correzione legata al passo temporale.

Restituisce un segnale deconvoluto della stessa lunghezza di `trace`.

#### `graham_smith_alg(mat)`

Esegue l'ortonormalizzazione di Gram-Schmidt su vettori disposti per righe in una matrice.

Restituisce un dizionario con:

- `new basis`: matrice con righe ortonormalizzate;
- `alphas`: norme dei vettori intermedi prima della normalizzazione.

### Analisi e preparazione di segnali 1D

#### `patcher(trace, tvec, d='auto', atol=1e-10)`

Estende un segnale all'inizio e alla fine con patch costruite tramite mezze onde sinusoidali al quadrato.

Serve a rendere il segnale piu stabile in procedure basate su FFT o deconvoluzione, riducendo discontinuita ai bordi.

Restituisce:

- il nuovo vettore temporale;
- il segnale patchato;
- gli estremi temporali del segmento originale dentro il nuovo segnale.

#### `extr_finder(trace)`

Trova gli estremi locali di un segnale 1D.

Restituisce:

- indici di tutti gli estremi;
- indici dei massimi locali;
- indici dei minimi locali.

#### `rel_peaks_finder(trace, delta, toll=0, return_extremant_indiced=False)`

Trova i picchi rilevanti di un segnale in base a una soglia di ampiezza `delta` rispetto ai minimi vicini.

- `toll` permette di includere picchi secondari vicini a picchi rilevanti.
- Con `return_extremant_indiced=True` restituisce anche gli estremi calcolati da `extr_finder`.

#### `add_section(start, stop, mask)`

Funzione ausiliaria che imposta a `True` una sezione di una maschera booleana tra due indici.

E usata internamente da `relevant_peaks_mask`.

#### `relevant_peaks_mask(trace, delta, toll=0, include_mins=True, return_mins=False)`

Identifica i picchi rilevanti e costruisce una maschera booleana che marca le regioni significative attorno a quei picchi.

- `include_mins=True` mantiene i minimi nella maschera.
- `return_mins=True` restituisce anche gli indici dei minimi, utile per evitare ricalcoli.

Restituisce i picchi rilevanti e la maschera booleana.

#### `Interp_grid(trace, Interpeak_mask, tvec, grid_spacing=5, correct_for_mins_duplication=False, minima_indices=None)`

Costruisce una griglia temporale regolare e calcola, per ogni nodo, il valore medio del segnale nelle finestre adiacenti usando solo i punti selezionati da `Interpeak_mask`.

- Richiede un vettore temporale uniformemente campionato.
- Scarta i nodi per cui non ci sono punti validi nella maschera.
- Puo correggere duplicazioni di minimi isolate nella griglia interpolata.

Restituisce la griglia temporale e i valori medi interpolati.

### Filtraggio

#### `absolute_grid_sum(trace, tvec, grid_spacing=0.5, return_grid=False)`

Divide il segnale in finestre definite da una griglia temporale e somma il valore assoluto delle somme locali.

E usata come misura della distanza tra un segnale originale e un segnale filtrato.

Con `return_grid=True` restituisce anche la griglia temporale e i contributi locali.

#### `Good_filter_index(Bessel_filtr_trace, original_trace, tvec, grid_spacing=0.5, N_m='auto', skin_param='auto', weight=100)`

Calcola un indice di qualita per un segnale filtrato.

L'indice combina:

- distanza dal segnale originale misurata su griglia;
- penalita per oscillazioni spurie, stimata tramite il numero di estremi.

Valori piu bassi indicano un filtro migliore.

#### `optimize_filter(trace_to_filter, tvec, bounds, order=8, grid_spacing=0.5, weight=100, skin_param='auto', return_filtered=True)`

Cerca la frequenza di cutoff ottimale per un filtro passa-basso di Bessel.

- `bounds` definisce l'intervallo di ricerca della frequenza di cutoff.
- `order` e l'ordine del filtro.
- Usa `Good_filter_index` per scegliere la frequenza migliore.
- Con `return_filtered=True` restituisce anche il segnale filtrato.

### Equazioni differenziali

#### `Runge_Kutta_4(r0, f, dt, T)`

Integra un sistema di equazioni differenziali ordinarie del primo ordine con il metodo Runge-Kutta classico del quarto ordine.

- `r0` e la condizione iniziale.
- `f` e la funzione che restituisce la derivata dello stato.
- `dt` e il passo temporale.
- `T` e il tempo finale.

Restituisce un dizionario con il vettore temporale e la traiettoria integrata.
