# Basic tools GM

Repository contenente un piccolo modulo Python, `basic_tools_GM/module.py`, con funzioni di utilita per analisi numerica, processi stocastici e gestione di dati tabulari.

## Dipendenze

Il modulo usa:

- `numpy`
- `scipy`

Nel file `module.py` e presente anche l'import di `datetime`, usato per aggiungere data e ora nei file di testo generati.

## Come installare il modulo

Da bash, puoi installare il modulo direttamente dalla repository GitHub con:

```bash
pip install git+https://github.com/giacomomoriondo2002/Basic_tools_GM.git
```

## Contenuto di `module.py`

### `autocorr_matrix_1D_system(X, N_iter_axis=0, unbiased=True)`

Calcola la matrice di autocovarianza a due tempi di un processo stocastico 1D a partire da un insieme di traiettorie.

- `X` contiene le realizzazioni del processo.
- `N_iter_axis` indica quale asse rappresenta le traiettorie indipendenti.
- `unbiased=True` usa il fattore di normalizzazione `N-1`; con `False` usa `N`.

Restituisce una matrice `(n_time, n_time)` in cui ogni elemento rappresenta la covarianza tra due istanti temporali.

### `Random_jump(W, current_state, State_vec, test_W=True)`

Estrae casualmente il prossimo stato di un processo di Markov discreto usando una matrice di transizione `W`.

- `W[i, j]` e la probabilita di passare dallo stato `j` allo stato `i`.
- `current_state` e lo stato attuale.
- `State_vec` contiene tutti gli stati possibili, scalari o vettoriali.
- `test_W=True` verifica che ogni colonna di `W` sommi a 1.

Restituisce un dizionario con l'indice del nuovo stato e il valore dello stato estratto.

### `txt_file_writer(filename, names_of_arrays, *arrays, digit_of_precision='all', mode='f', date=True, log_file=None, delimiter='\t', auto_ordering=True)`

Scrive piu array come colonne allineate in un file di testo.

Funzionalita principali:

- gestisce array di lunghezze diverse aggiungendo valori mancanti;
- scrive una riga di intestazione con i nomi delle colonne;
- puo includere data, ora e una descrizione dei dati;
- permette di scegliere precisione numerica, formato e delimitatore;
- puo ordinare automaticamente le colonne per lunghezza decrescente.

Restituisce `True` se la scrittura va a buon fine.

### `couple_sorting(x, y, descending=False)`

Ordina due sequenze in parallelo usando `x` come riferimento per l'ordinamento.

- `x` viene ordinato.
- `y` viene riordinato seguendo gli stessi indici di `x`.
- `descending=True` ordina in ordine decrescente.

Restituisce la coppia `(x_sorted, y_sorted)`.

### `best_index_finder(vec, *args, toll='auto', return_vec='best', mode='strict', ignore_length_warning=False)`

Trova gli indici di `vec` piu vicini a uno o piu valori target.

Modalita principali:

- `return_vec='best'`: restituisce l'indice del punto piu vicino per ogni target.
- `return_vec='all'`: restituisce tutti gli indici entro la tolleranza.
- `toll='auto'`: imposta la tolleranza come distanza media tra punti consecutivi di `vec`.
- `mode='strict'`: genera errore se un target non e abbastanza vicino.
- `mode='only close points'`: ignora i target troppo lontani.
- `mode='loose'`: restituisce sempre il punto piu vicino.

E utile per associare valori continui a una griglia discreta di punti.
