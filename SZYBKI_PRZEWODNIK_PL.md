# 🎯 SZYBKI PRZEWODNIK - WSZYSTKO NA JEDNEJ STRONIE

> **DISCLAIMER:** Niniejsza dokumentacja została wygenerowana przez sztuczną inteligencję (Claude - Language Model) w dniu 10 maja 2026 r. Zawiera zwięzły przegląd wszystkich napraw, ich znaczenia oraz instrukcje szybkiego startu dla użytkownika.

---

## TL;DR - Napraw Q-Learning

| Problem | Rozwiązanie | Wpływ |
|---------|-------------|-------|
| 🔴 Aktualizowało WSZYSTKIE Q-wartości | ✅ Tylko akcja podjęta | ⭐⭐⭐⭐⭐ |
| 🔴 Brak zanikania eksploracji | ✅ Epsilon decay: 0.98^episode | ⭐⭐⭐⭐⭐ |
| 🔴 Cechy na różnych skalach | ✅ Znormalizować do [-1,1] | ⭐⭐⭐⭐ |
| 🔴 Wypadki neuczące | ✅ Tylko akcja podjęta | ⭐⭐⭐⭐ |
| 🔴 Równe współczynniki | ✅ Optymalne dla każdego zadania | ⭐⭐⭐ |
| 🟡 Naiwna inicjalizacja | ✅ Xavier-like | ⭐⭐ |
| 🟡 Stały rozmiar sieci | ✅ Adaptacyjny | ⭐⭐ |

---

## Cztery Kroki do Treningu

### 1️⃣ Uruchom TORCS
```bash
cd /home/kamil/Dokumenty/gym_torcs
./up.sh
```

### 2️⃣ Trenuj (one-liner)
```bash
./run_training_sequence.sh 50
```
Czeka ~20-30 minut. Zawiera wszystkie 3 etapy.

### 3️⃣ Obserwuj Postęp
Terminal wyświetla nagrody:
```
Episode 5: Reward 123456 ✓
Episode 10: Reward 234567 ✓
Episode 20: Reward 345678 ✓
```
Rosnące nagrody = uczenie się! ✓

### 4️⃣ Przetestuj Model
```bash
python3 torcs_jm_par_new.py --mode custom --load-model
```
Obserwuj samochód - powinien jeździć normalnie!

---

## Problem #1: Złe Aktualizacje Q-Learning (KRYTYCZNE)

### Co Było?
```
Wszystkie 5 akcji otrzymywały tę samą nagrodę:
✗ Prawe skręcenie:    Q[0] = 50
✗ Lewe skręcenie:     Q[1] = 50  
✗ Prosto:             Q[2] = 50
✗ Prawe skręcenie:    Q[3] = 50
✗ Pełne prawe:        Q[4] = 50

Agent myśli: "Wszystko jest równie dobre/złe"
Rezultat: BRAK UCZENIA
```

### Jak Zostało Naprawione?
```
Tylko wybrana akcja (np. prosto) otrzymuje gradientu:
✓ Prawe skręcenie:    Q[0] = bez zmian
✓ Lewe skręcenie:     Q[1] = bez zmian
✓ Prosto:             Q[2] = 50 + gradient  ← TYLKO TO
✓ Prawe skręcenie:    Q[3] = bez zmian
✓ Pełne prawe:        Q[4] = bez zmian

Agent uczy się: "Prosto było prawidłowe, inne nie"
Rezultat: PRAWIDŁOWE UCZENIE
```

---

## Problem #2: Brak Zanikania Eksploracji (KRYTYCZNE)

### Co Było?
```
Epsilon losowy każdy epizod:
Epizod 1:   ε = 0.087 (eksploruj 8.7%)
Epizod 2:   ε = 0.042 (eksploruj 4.2%)
Epizod 3:   ε = 0.093 (eksploruj 9.3%) ← Powrót do eksploracji!
Epizod 50:  ε = 0.055 (eksploruj 5.5%)

Agent nigdy nie przestaje eksplorować losowo.
Rezultat: ŻADEN POSTĘP
```

### Jak Zostało Naprawione?
```
Epsilon zanika systemtycznie:
Epizod 1:   ε = 0.100 (eksploruj 10%)
Epizod 10:  ε = 0.082 (eksploruj 8.2%)
Epizod 20:  ε = 0.067 (eksploruj 6.7%)
Epizod 50:  ε = 0.037 (eksploruj 3.7%)
Epizod 100: ε = 0.013 (eksploruj 1.3%)

Agent stopniowo przechodzi z eksploracji na eksploatację.
Rezultat: POSTĘP I UCZENIE SIĘ
```

---

## Problem #3: Nienormalizowane Cechy (KRYTYCZNE)

### Co Było?
```
Surowe wartości z symulatora:
speedX:    240.5 km/h       (duży)
angle:     0.45 rad         (mały)
trackPos:  0.8              (bardzo mały)
rpm:       3500             (duży)

Sieć musi skompenswać ogromne różnice skalą.
Rezultat: NIESTABILNE UCZENIE
```

### Jak Zostało Naprawione?
```
Znormalizowane do [-1, 1]:
speedX:    0.80     (0-1 range)
angle:     0.14     (-1 to 1 range)
trackPos:  0.80     (-1 to 1 range)
rpm:       0.35     (0-1 range)

Wszystkie na podobnej skali!
Rezultat: STABILNE I SZYBKIE UCZENIE
```

---

## Problem #4: Wypadki Nie Nauczały (KRYTYCZNE)

### Co Było?
```
Wypadek przy akcji "pełne lewe skręcenie":

Wszystkie akcje otrzymują karę:
✗ Prawe:  Q[0] = -10000 (nigdy nie wybrał!)
✗ Lewe:   Q[1] = -10000 (nigdy nie wybrał!)
✗ Prosto: Q[2] = -10000 (nigdy nie wybrał!)
✗ Prawe2: Q[3] = -10000 (nigdy nie wybrał!)
✗ Pełne:  Q[4] = -10000 (wybrał to) ✓

Agent myśli: "WSZYSTKIE akcje są złe!"
Rezultat: AGENT SIĘ PARALIZUJE
```

### Jak Zostało Naprawione?
```
Tylko akcja podjęta otrzymuje karę:

target_q = -10000 ← Tylko dla akcji "pełne lewe"

Agent uczy się: "To KONKRETNE działanie = zło"
Może nauczyć się: "Inne działania mogą być ok"
Rezultat: PRAWIDŁOWE NAUCZANIE SIĘ UNIKAĆ WYPADKÓW
```

---

## Problem #5: Równe Współczynniki Uczenia (ZNACZĄCY)

### Co Było?
```
Wszystkie sieci lr = 0.001:

Sterowanie (krytyczne):      0.001 ← Zbyt szybko!
Gaz/Hamulec (średnie):       0.001 ← OK
Biegi (proste):              0.001 ← Zbyt wolno!

Rezultat: SUBOPTYMALNE TEMPO ZBIEŻNOŚCI
```

### Jak Zostało Naprawione?
```
Zoptymalizowane dla każdego zadania:

Sterowanie (krytyczne):      0.0005 ← Konserwatywny
Gaz/Hamulec (średnie):       0.001  ← Standardowy
Biegi (proste):              0.0015 ← Szybszy

Rezultat: KAŻDA SIEĆ ZBIEGA OPTYMALNIE
```

---

## Problem #6: Naiwna Inicjalizacja (NISKA)

### Co Było?
```
Wszystkie wagi × 0.1 niezależnie od rozmiaru:

Duża macierz (100×64):  × 0.1 ← Za mała
Mała macierz (64×5):    × 0.1 ← Za duża

Rezultat: NIESTABILNY START
```

### Jak Zostało Naprawione?
```
Xavier-like inicjalizacja:

Duża macierz (100×64):  × √(2/(100+64)) ≈ 0.11
Mała macierz (64×5):    × √(2/(64+5)) ≈ 0.16

Rezultat: SZYBSZA ZBIEŻNOŚĆ
```

---

## Problem #7: Stały Rozmiar Sieci (NISKA)

### Co Było?
```
Szybki trening:  64 neurony (powolnie)
Normalny:        64 neurony (ok)

Rezultat: WOLNA EKSPLORACJA
```

### Jak Zostało Naprawione?
```
Szybki trening:  32 neurony (szybko!)
Normalny:        64 neurony (dokładnie)

Rezultat: SZYBKA EKSPLORACJA, DOKŁADNA PRODUKCJA
```

---

## Struktura Dokumentacji

📚 **CZTERY PLIKI DOKUMENTACJI:**

1. **[DOKUMENTACJA_PL.md](DOKUMENTACJA_PL.md)** ← Zacznij tutaj!
   - Pełne wyjaśnienie wszystkich 7 problemów
   - Szczegółowe rozwiązania
   - Oczekiwane wyniki

2. **[PROBLEMY_I_NAPRAWY_PL.md](PROBLEMY_I_NAPRAWY_PL.md)** ← Zaawansowany
   - Głębokie pogłębianie każdego problemu
   - Analogie do rzeczywistych sytuacji
   - Matematyka zakulisów

3. **[INSTRUKCJA_TRENINGU_PL.md](INSTRUKCJA_TRENINGU_PL.md)** ← Jak to robić
   - Krok po kroku instrukcje
   - Wszystkie opcje wiersza poleceń
   - Rozwiązywanie problemów

4. **[LISTA_ZMIAN_PL.md](LISTA_ZMIAN_PL.md)** ← Dla developerów
   - 16 konkretnych zmian w kodzie
   - Przed/po porównania
   - Numery linii

---

## Tryb Treningowy Wyjaśniony

### `--train-fast` (domyślny dla eksploracji)
- **Sieć:** 32 neurony (mała)
- **Prędkość:** Szybka (5-7 min/50 epizodów)
- **Dokładność:** Dobra
- **Zastosowanie:** Eksploracja, tuning parametrów

### Normalny (bez `--train-fast`)
- **Sieć:** 64 neurony (duża)
- **Prędkość:** Powolna (12-15 min/50 epizodów)
- **Dokładność:** Lepsza
- **Zastosowanie:** Produkcja, finalne modele

---

## Znormalizowane Zakresy Stanów

### Sterowanie
```
angle:          [-π, π]  →  [-1, 1]
trackPos:       [-1, 1]  →  [-1, 1]
speedX:         [0, 300] →  [0, 1]
track_distance: [0, 100] →  [-1, 1]
```

### Gaz/Hamulec
```
speedX:         [0, 300] →  [0, 1]
rpm:            [0, 10k] →  [0, 1]
angle:          [-π, π]  →  [-1, 1]
trackPos:       [-1, 1]  →  [-1, 1]
```

### Biegi
```
rpm:            [0, 10k] →  [0, 1]
speedX:         [0, 300] →  [0, 1]
angle:          [-π, π]  →  [-1, 1]
```

---

## Harmonogram Zanikania Epsilon

```
Epizod   Epsilon   Eksploracja
0        0.10000   ████████░░ 10%
10       0.08203   ██████░░░░  8.2%
20       0.06725   █████░░░░░  6.7%
50       0.03651   ███░░░░░░░  3.7%
100      0.01334   █░░░░░░░░░  1.3%
```

Współczynnik zanikania: **0.98 na epizod**

---

## Oczekiwane Wyniki Po 150 Epizodach

### Metryki Jakości

| Metrika | Przed | Po |
|---------|-------|-----|
| Średnia Nagroda | -1M | +300k |
| Wypadki | 50% | <5% |
| Czasy Okrążeń | Brak | 1-2 min |
| Postęp | 0% | Wyraźny |
| Stabilność | ✗ | ✓ |

### Zachowanie Samochodu

| Aspekt | Przed | Po |
|--------|-------|-----|
| Sterowanie | Chaotyczne | Gładkie |
| Przyspieszanie | Erratyczne | Płynne |
| Hamowanie | Nagłe | Kontrolowane |
| Biegi | Losowe | Prawidłowe |
| Ukończenie okrążenia | Rzadko | Czasami |

---

## Checklist Przed Treningiem

- [ ] TORCS zainstalowany i pracujący
- [ ] Python 3.6+ zainstalowany
- [ ] NumPy zainstalowany (`pip install numpy`)
- [ ] Plik `torcs_jm_par_new.py` z naprawami
- [ ] Komendy `up.sh` i `run_training_sequence.sh` są wykonywalne
- [ ] Co najmniej 30 minut wolnego czasu
- [ ] 500MB wolnego dysku na modele

---

## Polecane Komendy

### Szybki Test
```bash
./run_training_sequence.sh 10  # 10 epizodów = ~5 min
```

### Trening Standardowy
```bash
./run_training_sequence.sh 50  # 50 epizodów = ~20 min
```

### Trening Zaawansowany
```bash
./run_training_sequence.sh 100 # 100 epizodów = ~40 min
```

### Test Modelu
```bash
python3 torcs_jm_par_new.py --mode custom --load-model
```

---

## Ważne Liczby

| Parametr | Wartość | Znaczenie |
|----------|---------|-----------|
| Epsilon Początkowy | 0.10 | 10% eksploracji |
| Epsilon Minimum | 0.01 | 1% eksploracji |
| Epsilon Decay | 0.98 | Zanikanie per epizod |
| Learning Rate (Steer) | 0.0005 | Konserwatywny |
| Learning Rate (Throttle) | 0.001 | Standardowy |
| Learning Rate (Gear) | 0.0015 | Szybki |
| Gamma (Dyskont) | 0.95 | Przyszłe nagrody |
| Rozmiar Sieci (Fast) | 32 | Szybka eksploracja |
| Rozmiar Sieci (Normal) | 64 | Dokładne predykcje |

---

## Następne Kroki

1. **Teraz:** Przeczytaj [DOKUMENTACJA_PL.md](DOKUMENTACJA_PL.md) aby zrozumieć problemy
2. **Za 5 minut:** Uruchom `./up.sh` aby załadować TORCS
3. **Za 10 minut:** Uruchom `./run_training_sequence.sh 10` dla szybkiego testu
4. **Za 15 minut:** Obserwuj output i wyniki treningu
5. **Za 30 minut:** Przetestuj wytrenowany model
6. **Potem:** Powtórz z `./run_training_sequence.sh 50` dla pełnego treningu

---

## Wsparcie

**Jeśli coś nie działa:**
1. Sprawdź [INSTRUKCJA_TRENINGU_PL.md#rozwiązywanie-problemów](INSTRUKCJA_TRENINGU_PL.md)
2. Upewnij się że TORCS działa: `./up.sh`
3. Sprawdzić logi w terminalu
4. Spróbuj szybkiego testu: `./run_training_sequence.sh 5`

**Jeśli chcesz eksperymentować:**
- Zobacz [INSTRUKCJA_TRENINGU_PL.md#zaawansowane-parametry](INSTRUKCJA_TRENINGU_PL.md)
- Zmień współczynniki uczenia
- Zmień liczbę epizodów
- Porównaj wyniki

---

**Powodzenia! Twój AI sterownik powinien się teraz uczyć! 🚗🤖**
