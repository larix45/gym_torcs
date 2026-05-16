# 🚀 INSTRUKCJA TRENINGU - PORADNIK PRAKTYCZNY

> **DISCLAIMER:** Niniejsza dokumentacja została wygenerowana przez sztuczną inteligencję (Claude - Language Model) w dniu 10 maja 2026 r. Zawiera szczegółowe instrukcje krokowe dotyczące treningu sztucznej inteligencji w symulatorze TORCS przy użyciu naprawionego algorytmu Q-Learning.

---

## Szybki Start (3 Komendy)

Jeśli tylko chcesz szybko uruchomić trening:

```bash
cd /home/kamil/Dokumenty/gym_torcs

# Uruchom pełną sekwencję treningu
./run_training_sequence.sh 50
```

To:
1. Rozpoczyna TORCS
2. Trenuje sterowanie przez 50 epizodów
3. Trenuje gaz/hamulec przez 50 epizodów
4. Trenuje biegi przez 50 epizodów
5. Zapisuje wszystkie wytrenowane modele

Czekaj ~20-30 minut. Gotowe!

---

## Trening Etap po Etapie (Zaawansowane)

Jeśli chcesz kontrolować każdy etap ręcznie:

### Krok 1: Upewnij się że TORCS Działa

```bash
# Uruchom upami skrypt (otwiera TORCS i VS Code)
cd /home/kamil/Dokumenty/gym_torcs
./up.sh
```

Czekaj że TORCS się załaduje (widoczne będzie okno symulatora).

### Krok 2: Etap 1 - Trening Sterowania

W nowym oknie terminala:

```bash
cd /home/kamil/Dokumenty/gym_torcs

python3 torcs_jm_par_new.py \
  --train \
  --train-fast \
  --mode qlearn \
  --episodes 50 \
  --train-steer \
  --fixed-throttle \
  --fixed-gear \
  --save-model \
  --steer-model-file steer.npz
```

**Co się dzieje:**
- Agent uczy się sterowania przez 50 epizodów
- Gaz i biegi są stałe (niezmienne)
- Nagrody wyświetlane co kilka epizodów
- Po każdym epizodzie sieć się uczy

**Oczekiwany rezultat:**
```
Epizod 0:   Reward: -523,451 (losowe skręcanie)
Epizod 5:   Reward: -12,340 (już lepiej)
Epizod 10:  Reward: 34,200 (wyraźnie się uczy)
Epizod 30:  Reward: 456,789 (dobre sterowanie)
Epizod 50:  Reward: 567,890 (najlepsze sterowanie)
```

**Czas trwania:** ~7-10 minut

### Krok 3: Etap 2 - Trening Gazu/Hamulca

```bash
python3 torcs_jm_par_new.py \
  --train \
  --train-fast \
  --mode qlearn \
  --episodes 50 \
  --train-throttle \
  --load-model \
  --steer-model-file steer.npz \
  --fixed-gear \
  --save-model \
  --throttle-model-file throttle.npz
```

**Kluczowe opcje:**
- `--load-model`: Ładuje wytrenowaną sieć sterowania ze steer.npz
- `--steer-model-file steer.npz`: Ścieżka do modelu sterowania
- `--train-throttle`: Uczy się tylko gazu/hamulca
- `--fixed-gear`: Biegi pozostają na stałej wartości

**Oczekiwany rezultat:**
- Czystsze przyspieszanie i hamowanie
- Lepsze utrzymanie prędkości docelowej
- Bardziej płynne obsługi hamowania na zakrętach

**Czas trwania:** ~7-10 minut

### Krok 4: Etap 3 - Trening Biegów

```bash
python3 torcs_jm_par_new.py \
  --train \
  --train-fast \
  --mode qlearn \
  --episodes 50 \
  --train-gear \
  --load-model \
  --steer-model-file steer.npz \
  --throttle-model-file throttle.npz \
  --save-model \
  --gear-model-file gear.npz
```

**Kluczowe opcje:**
- Ładuje zarówno model sterowania jak i gazu/hamulca
- `--train-gear`: Uczy się tylko przerzucania biegów
- Biegi są teraz zmienną, a nie stałą

**Oczekiwany rezultat:**
- Prawidłowe przerzucanie biegów
- Obroty w efektywnym zakresie
- Płynniejsza dostawa mocy

**Czas trwania:** ~5-7 minut

---

## Test Wytrenowanego Modelu

Po wytrenowaniu wszystkich trzech etapów, przetestuj pełny model:

```bash
python3 torcs_jm_par_new.py \
  --mode custom \
  --load-model \
  --steer-model-file steer.npz \
  --throttle-model-file throttle.npz \
  --gear-model-file gear.npz
```

**Bez flagii `--train`** - agent nie uczy się, tylko używa wytrenowanych wag.

### Obserwacje w Oknie Symulatora

**Dobre znaki:**
- Samochód porusza się gładko
- Minimalne oscylacje kierownice
- Stabilny wibór biegu
- Płynne przyspieszanie/hamowanie
- Okrążenie może być ukończone

**Złe znaki:**
- Szybkie oscylacje kierownicy
- Nienaturalne przeskokami prędkości
- Częste wypadki
- Wychodzenie z drogi

---

## Monitorowanie Treningu

### Główne Metryką: Nagroda (Reward)

Każdy epizod wyświetla coś takiego:

```
Episode 5, Steer - Total Reward: 123456.78
Episode 6, Steer - Total Reward: 145678.90
Episode 7, Steer - Total Reward: 156789.23
```

**Interpretacja:**
- Rosnące nagrody = uczenie się przebiega pomyślnie ✓
- Oscylujące nagrody = normalne, eksploracja i eksploatacja
- Malejące nagrody = może być problem

### Średnia Nagrody

Po każdych 10 epizodach skrypt wyświetla średnią:

```
Average Reward (last 10 episodes): 234567.89
```

**Docelowe wartości:**
- Sterowanie: 300,000 - 500,000
- Gaz/Hamulec: 200,000 - 400,000
- Biegi: 100,000 - 250,000

(Wartości różnią się w zależności od charakteru toru i konfiguracji)

### Wskaźnik Wypadków

```
Crashes: 3 out of 50 episodes (6%)
```

**Dobre wartości:**
- Etap 1 (sterowanie): < 20%
- Etap 2 (gaz): < 10%
- Etap 3 (biegi): < 5%

---

## Zaawansowane Parametry

### Zmiana Liczby Epizodów

```bash
./run_training_sequence.sh 100  # 100 epizodów na każdy etap
./run_training_sequence.sh 20   # 20 epizodów na każdy etap (szybki test)
```

### Tryb Normalne (zamiast Szybkiego)

Domyślnie `--train-fast` używa małych sieci (32 neurony).

Bez flagi `--train-fast` używane będą pełne sieci (64 neurony):

```bash
python3 torcs_jm_par_new.py \
  --train \
  --mode qlearn \
  --episodes 50 \
  --train-steer \
  --save-model \
  --steer-model-file steer_full.npz
```

**Porównanie:**
| Cecha | Fast | Normal |
|-------|------|--------|
| Rozmiar sieci | 32 neurony | 64 neurony |
| Prędkość | Szybka | Wolna |
| Dokładność | Dobra | Lepsza |
| Pamięć | Mała | Większa |
| Idealne do | Eksploracji | Produkcji |

### Zmiana Współczynnika Uczenia

W kodzie znajdź:

```python
steer_lr = 5e-4       # Zmień na np. 1e-3
throttle_lr = 1e-3    # Zmień na np. 2e-3
gear_lr = 1.5e-3      # Zmień na np. 3e-3
```

Wyższy współczynnik = szybsze uczenie, ale mniej stabilne.

### Zmiana Zanikania Epsilon

```python
EPSILON_DECAY_RATE = 0.98  # Zmień na np. 0.95 (szybsze zanikanie)
```

Wyższe wartości = wolniejsze zanikanie eksploracji.

---

## Rozwiązywanie Problemów

### Problem: Nagrody Nie Rosną

**Przyczyny:**
1. TORCS nie jest uruchomiony
2. Sieci zbyt duże (offline learning)
3. Współczynnik uczenia za wysoki (oscylacje)
4. Współczynnik uczenia za niski (bez postępu)

**Rozwiązanie:**
```bash
# Upewnij się że TORCS działa
./up.sh

# Zrestartuj trening
./run_training_sequence.sh 10  # Krótki test
```

### Problem: Samochód Się Nie Rusza

**Przyczyny:**
1. Model nie został załadowany prawidłowo
2. Brak współpracy między etapami

**Rozwiązanie:**
```bash
# Sprawdź czy pliki modeli istnieją
ls -lah *.npz

# Powinny być: steer.npz, throttle.npz (lub .npy)
```

### Problem: Bardzo Powoli Trenuję

**Przyczyny:**
1. Używasz normalnego trybu zamiast --train-fast
2. Za duża liczba epizodów
3. Słaby komputer

**Rozwiązanie:**
```bash
# Zawsze używaj --train-fast
./run_training_sequence.sh 10  # Mniej epizodów do testowania
```

---

## Zapis i Wczytywanie Modeli

### Automatyczne Zapisywanie

Flaga `--save-model` automatycznie zapisuje modele:

```bash
# Zapisze do steer.npz po treningu
--train-steer --save-model --steer-model-file steer.npz
```

### Ręczne Zapisywanie

Jeśli chcesz zachować stary model:

```bash
cp steer.npz steer_backup.npz
# Teraz nowy trening będzie nadpisywać tylko steer.npz
```

### Wczytywanie Różnych Modeli

```bash
python3 torcs_jm_par_new.py \
  --mode custom \
  --load-model \
  --steer-model-file /path/to/steer_old.npz \
  --throttle-model-file /path/to/throttle_old.npz \
  --gear-model-file /path/to/gear_old.npz
```

---

## Eksperymentowanie i Iteracja

### Eksperyment 1: Porównanie Współczynników Uczenia

```bash
# Szybki (lr=1e-3)
# ... trening ...

cp steer.npz steer_fast_lr.npz

# Powolny (lr=1e-4)
# Zmień w kodzie lr = 1e-4
# ... trening ...

cp steer.npz steer_slow_lr.npz
```

Porównaj wyniki testując oba modele.

### Eksperyment 2: Liczba Epizodów

```bash
# Mało epizodów (10)
./run_training_sequence.sh 10
cp *.npz results_10_episodes/

# Średnio epizodów (50)
./run_training_sequence.sh 50
cp *.npz results_50_episodes/

# Wiele epizodów (100)
./run_training_sequence.sh 100
cp *.npz results_100_episodes/
```

Porównaj jakość jazdy dla każdego.

---

## Pełna Dokumentacja Opcji

```
Opcje Ogólne:
  --train                Train mode (trainuj sieć)
  --train-fast          Use smaller networks (32 hidden units)
  --mode qlearn         Q-learning mode
  --mode custom         Use pre-trained models only
  --mode modular        Use hardcoded rules

Opcje Treningu:
  --episodes NUM        Liczba epizodów do treningu
  --train-steer         Trenuj sieć sterowania
  --train-throttle      Trenuj sieć gazu/hamulca
  --train-gear          Trenuj sieć biegów
  --fixed-throttle      Fiksuj gaz na domyślną wartość
  --fixed-gear          Fiksuj biegi na domyślną wartość

Opcje Modeli:
  --load-model          Ładuj istniejące modele
  --save-model          Zapisz modele po treningu
  --steer-model-file    Ścieżka do modelu sterowania
  --throttle-model-file Ścieżka do modelu gazu
  --gear-model-file     Ścieżka do modelu biegów
```

---

## Oczekiwane Czasy

| Etap | Epizodów | Czas | Tryb |
|------|----------|------|------|
| Sterowanie | 50 | 7-10 min | Fast |
| Gaz/Hamulec | 50 | 7-10 min | Fast |
| Biegi | 50 | 5-7 min | Fast |
| **Razem** | **150** | **19-27 min** | **Fast** |
| | | | |
| Sterowanie | 50 | 12-15 min | Normal |
| Gaz/Hamulec | 50 | 12-15 min | Normal |
| Biegi | 50 | 10-12 min | Normal |
| **Razem** | **150** | **34-42 min** | **Normal** |

---

## Najlepsze Praktyki

1. **Zawsze trenuj w kolejności:** Sterowanie → Gaz → Biegi
   - Nie odwracaj kolejności
   - Każdy etap zależy od poprzedniego

2. **Zawsze używaj `--train-fast` podczas eksploracji**
   - Szybciej iterujesz
   - Eksperymentujesz z parametrami

3. **Po znalezieniu dobrych parametrów, trenuj z `--normal`**
   - Większe sieci = lepsze wyniki

4. **Zawsze weryfikuj wyniki testując model**
   - Nie zakładaj że wyższe nagrody = lepsze jeżdżenie

5. **Zapisuj wersje modeli**
   - `steer_v1.npz`, `steer_v2.npz`, itp.
   - Możliwe porównanie wersji

---

## Następne Kroki

1. Uruchom szybki test:
   ```bash
   ./run_training_sequence.sh 10
   ```

2. Obserwuj output w terminalu

3. Przetestuj wytrenowany model:
   ```bash
   python3 torcs_jm_par_new.py --mode custom --load-model
   ```

4. Jeśli ok, trenuj z większą liczbą epizodów:
   ```bash
   ./run_training_sequence.sh 50
   ```

Powodzenia! 🎉
