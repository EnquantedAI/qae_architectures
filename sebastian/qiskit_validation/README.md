# QuTSAE — rozszerzenie ICCS 2024 → CMES (pakiet do uruchomienia na GPU)

Ten katalog to samodzielny pakiet do dokończenia obliczeń rozszerzenia
artykułu *"Design Considerations for Denoising Quantum Time Series
Autoencoder"* (Cybulski & Zając, ICCS 2024) na potrzeby numeru specjalnego
CMES *"Quantum Machine Learning: Methods and Engineering Applications"*
(termin: 30.09.2026).

Przygotowany w sesji chmurowej (2 rdzenie CPU, brak GPU) — stąd `qutsae.py`
jest sportowany do nowoczesnego Qiskit (2.5.x), bo oryginalne notebooki z
repo `ironfrown/ts_anomaly_detection_by_denoising` używają usuniętego już
`qiskit.Aer` / prymitywów Sampler V1. Wszystko poniżej było testowane na
CPU; ścieżka GPU (`device='GPU'` w `qutsae.py` / `cuda` w `classical_ae.py`)
jest zaimplementowana i powinna działać automatycznie, ale nie mogłem jej
zweryfikować na realnym sprzęcie — stąd `quick_smoke_test.py` jako pierwszy
krok.

Docelowa maszyna: **Ubuntu, 2x NVIDIA GPU — A40 + Tesla L4, po 24GB+
GDDR6/GDDR6 każda**. Obie karty to sporo więcej pamięci i mocy niż potrzeba
dla symulacji 11-kubitowej (2048 amplitud, mikroskopijne jak na te karty) —
sekcja "Dwie karty równolegle" niżej pokazuje jak wykorzystać obie naraz.

## Zawartość

```
qutsae.py                        # rdzeń: encoding, ansatz (Ry i Rx+Ry+aw), pełny QAE, trening COBYLA
classical_ae.py                  # klasyczne baseline'y: MLP-AE, LSTM-AE
datasets_new.py                  # 3 nowe zbiory: energy (VIC electricity), finance (AAPL) - realne, nie-chaotyczne;
                                  # mackey_glass (tau=17) - syntetyczny, chaotyczny (kontrast do pozostałych)
utils/                           # oryginalne narzędzia z repo (TS.py, Window.py, Target.py, Callback.py)
data/                            # surowe CSV dla energy/finance (mackey_glass generowany w locie)
requirements.txt                 # wersje pakietów (+ warianty CPU/GPU)
quick_smoke_test.py              # URUCHOM TO PIERWSZE (1-3 min) - sprawdza GPU i szacuje czas pełnego grida
run_full_grid_gpu.py             # GŁÓWNY grid: lat=7,trash=1,aw=3,reps=2, 2000 epok, 4 zbiory x 5 ziaren
run_two_gpu.sh                   # uruchamia run_full_grid_gpu.py RÓWNOLEGLE na obu kartach (A40 + L4)
aggregate_results.py             # bezpiecznie zbiera wyniki z results/quantum_full/*.json w podsumowanie (dowolny moment, nawet w trakcie)
run_classical_grid.py            # klasyczne AE, pełna siatka lat=1..8 (już uruchomione, wyniki w results/)
run_classical_grid_multiseed.py  # klasyczne AE, lat=7, 5 ziaren (już uruchomione, wyniki w results/)
run_quantum_grid.py              # zredukowana wersja (aw=0) uruchomiona w sandboxie deweloperskim - punkt odniesienia
results/                         # już policzone wyniki (klasyczne w pełni; kwantowe - wersja zredukowana)
```

## Instalacja

```bash
nvidia-smi   # sprawdź że obie karty (A40 + L4) są widoczne i sterownik działa

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# requirements.txt ma domyślnie ustawioną linię pod GPU (qiskit-aer-gpu-cu11);
# jeśli wolisz najpierw sprawdzić na CPU, zamień ją na qiskit-aer==0.17.2
```

## Uruchomienie

```bash
# 1. Zawsze najpierw to (1-3 min) - potwierdza że GPU jest faktycznie widziane
python quick_smoke_test.py

# 2a. Jedna karta na raz (prostsze, ~2x wolniejsze niż 2b):
python run_full_grid_gpu.py

# 2b. OBIE karty równolegle (zalecane - masz dwie, więc wykorzystaj obie;
#     dzieli 5 ziaren na dwie grupy, po jednej na kartę):
chmod +x run_two_gpu.sh
./run_two_gpu.sh
tail -f logs/gpu0.log logs/gpu1.log     # podgląd postępu obu procesów

# Opcjonalnie (dot. 2a): pojedynczy zbiór / podzbiór ziaren / szybszy test
python run_full_grid_gpu.py --dataset energy
python run_full_grid_gpu.py --seeds 2023 2024 --epochs 200

# Skrypt jest wznawialny - jeśli przerwiesz i uruchomisz ponownie, pomija
# kombinacje (dataset, seed), dla których plik wynikowy już istnieje w
# results/quantum_full/. To samo dotyczy dwóch procesów uruchomionych
# równolegle przez run_two_gpu.sh - nie nadpisują sobie nawzajem plików,
# bo każda para (dataset, seed) ma swój własny plik wyjściowy.

# 3. W dowolnym momencie (także w trakcie liczenia) zbierz wyniki w jedno
#    podsumowanie + zobacz średnią +/- SD per zbiór danych:
python aggregate_results.py
```

Wyniki lądują w `results/quantum_full/{dataset}_seed{seed}.json` (każdy
przebieg = osobny plik, stąd bezpieczne pisanie równoległe z dwóch kart).
`python aggregate_results.py` zbiera je w `results/quantum_full/_summary.json`
i `_summary.csv` — uruchom go w dowolnym momencie, nie tylko na końcu.

## Co już jest policzone (w `results/`)

- `classical_grid.json` — pełna siatka klasyczna (MLP-AE, LSTM-AE) × lat=1..8 × 4 zbiory (beer/energy/finance/mackey_glass), 1 ziarno. **Gotowe.**
- `classical_grid_multiseed.json` — MLP-AE, LSTM-AE @ lat=7 × 5 ziaren × 4 zbiory, do porównania średnia±SD z gridem kwantowym. **Gotowe.**
- `quantum_grid.json` — QuTSAE w zredukowanej konfiguracji (aw=0, 8 kubitów, 1000 epok, 1 ziarno) na 3 zbiorach (bez mackey_glass — dodany później), liczone w sandboxie deweloperskim jako tymczasowy punkt odniesienia, zanim ten pakiet trafił na GPU. **Nie jest to configuracja z papieru** (papier używa aw=3) — traktować jako dolny szacunek jakości, docelowy wynik to `results/quantum_full/`.

### Nowość: `mackey_glass` (tau=17) jako czwarty zbiór

Na prośbę: dołączony syntetyczny zbiór chaotyczny (równanie Mackey-Glass,
tau=17 — klasyczny "próg chaosu" w literaturze, patrz `datasets_new.py` dla
szczegółów wraz z uzasadnieniem burn-in). To celowy kontrast do trzech
zbiorów nie-chaotycznych (beer/energy/finance) — pokazuje, czy QuTSAE
generalizuje też na reżim chaotyczny, bliższy oryginalnej motywacji tego typu
architektur (patrz book/part3/ch09_qae_pure.qmd w repo źródłowym, gdzie
Mackey-Glass tau=17 jest przykładem uczącym). Klasyczne baseline'y dla tego
zbioru są już policzone (patrz wyżej); kwantowy wynik czeka na
`run_full_grid_gpu.py` na Waszym GPU.

## Kontekst / dalsze kroki

Pełny plan rozszerzenia (harmonogram, uzasadnienie doboru zbiorów, struktura
artykułu wg wytycznych CMES) jest w dokumencie `Plan_rozszerzenia_CMES.docx`
przekazanym wcześniej w rozmowie. Po ukończeniu `run_full_grid_gpu.py`
kolejny krok to złożenie wyników w sekcję Results artykułu i porównanie
QuTSAE vs. MLP-AE vs. LSTM-AE na wszystkich trzech zbiorach.

## Co dodatkowo umożliwia ten sprzęt (opcjonalne, do decyzji)

Ustalony wcześniej zakres MVP (config z papieru, 3 zbiory x 5 ziaren) to
15 przebiegów — na A40/L4 (11 kubitów = 2048 amplitud, trywialne dla tych
kart) powinno to zejść z szacowanych 17h/przebieg (2-rdzeniowy CPU) do
zapewne kilku-kilkunastu minut/przebieg, czyli cały MVP realny w godziny,
nie dni. Zostaje więc spory zapas czasu do 30.09. Opcje "stretch" z planu
CMES, które ten zapas by pokrył (nieuruchamiane automatycznie — patrz plik
`Plan_rozszerzenia_CMES.docx`):

- przeskanowanie `aw`/`reps` jak Tabela 1 w oryginalnym papierze (np. aw=1..5,
  reps=1..4), zamiast tylko najlepszej konfiguracji — pokazuje trend, nie
  tylko punkt
- więcej ziaren (np. 10 zamiast 5) dla węższych przedziałów ufności
- symulacja z modelem szumu NISQ (nie tylko idealny symulator) — bliżej
  realnego QPU
- czwarty zbiór danych, jeśli chcecie jeszcze szerszej generalizacji

Żadna z tych opcji nie jest jeszcze zaimplementowana w tym pakiecie — to
pytanie otwarte, czy je dodawać, czy trzymać się ustalonego zakresu MVP.
