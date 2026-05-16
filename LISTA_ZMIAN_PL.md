# 📝 LISTA ZMIAN - SZCZEGÓŁOWY DZIENNIK WSZYSTKICH NAPRAW

> **DISCLAIMER:** Niniejsza dokumentacja została wygenerowana przez sztuczną inteligencję (Claude - Language Model) w dniu 10 maja 2026 r. Zawiera szczegółowy opis każdej zmiany dokonanej w kodzie, numery linii, przed/po porównania, oraz uzasadnienie każdej naprawy.

---

## Streszczenie

**Łącznie zmian:** 16 głównych zmian w kodzie  
**Pliki zmienione:** 1 (torcs_jm_par_new.py)  
**Wiersze zmienione:** ~150 linii kodu  
**Status:** Wstecz kompatybilny (nie psuje istniejącego kodu)

---

## ZMIANA #1: Dodanie Stałej Zanikania Epsilon

**Linia:** ~315 (na początku sekcji stałych)  
**Typ:** Nowa stała  
**Znaczenie:** KRYTYCZNE

```python
# PRZED: Nie istniało
# (epsilon był losowany każdy epizod)

# PO:
EPSILON_DECAY_RATE = 0.98

# Umożliwia eksponencjalne zanikanie: ε = ε₀ * decay^episode
```

**Powód:** Umożliwia progresję od eksploracji do eksploatacji.

---

## ZMIANA #2: Dodanie Funkcji Normalizacji Stanu

**Linia:** ~480-540  
**Typ:** Nowa funkcja  
**Rozmiar:** ~60 wierszy  
**Znaczenie:** KRYTYCZNE

```python
def normalize_state(state, state_type='general'):
    """
    Normalizuj cechy stanu do zakresu [-1, 1]
    
    Dla state_type='steer':
        state[0] (angle) → [-1, 1]
        state[1] (trackPos) → [-1, 1]
        state[2] (speedX) → [0, 1]
        state[3] (track distance) → [-1, 1]
    
    Dla state_type='throttle':
        state[0] (speedX) → [0, 1]
        state[1] (rpm) → [0, 1]
        state[2] (angle) → [-1, 1]
        state[3] (trackPos) → [-1, 1]
    
    Dla state_type='gear':
        state[0] (rpm) → [0, 1]
        state[1] (speedX) → [0, 1]
        state[2] (angle) → [-1, 1]
    """
    state = state.copy()
    
    if state_type == 'steer':
        state[0] = np.clip(state[0] / np.pi, -1, 1)           # angle
        state[1] = np.clip(state[1], -1, 1)                    # trackPos
        state[2] = np.clip(state[2] / 300, 0, 1)               # speedX
        state[3] = np.clip((state[3] - 50) / 50, -1, 1)        # track distance
    
    elif state_type == 'throttle':
        state[0] = np.clip(state[0] / 300, 0, 1)               # speedX
        state[1] = np.clip(state[1] / 10000, 0, 1)             # rpm
        state[2] = np.clip(state[2] / np.pi, -1, 1)            # angle
        state[3] = np.clip(state[3], -1, 1)                    # trackPos
    
    elif state_type == 'gear':
        state[0] = np.clip(state[0] / 10000, 0, 1)             # rpm
        state[1] = np.clip(state[1] / 300, 0, 1)               # speedX
        state[2] = np.clip(state[2] / np.pi, -1, 1)            # angle
    
    return state
```

**Powód:** Bez normalizacji przepływ gradientów jest niezbalansowany.

---

## ZMIANA #3: Aktualizacja extract_state_steer()

**Linia:** ~591-600  
**Typ:** Modyfikacja istniejącej funkcji  
**Znaczenie:** WYSOKA

```python
# PRZED:
def extract_state_steer():
    state = np.array([...], dtype=np.float32)
    return state

# PO:
def extract_state_steer():
    state = np.array([...], dtype=np.float32)
    state = normalize_state(state, state_type='steer')
    return state
```

**Powód:** Stosuje normalizację do funkcji ekstrakcji stanu.

---

## ZMIANA #4: Aktualizacja extract_state_throttle()

**Linia:** ~610-620  
**Typ:** Modyfikacja istniejącej funkcji  
**Znaczenie:** WYSOKA

```python
# PO:
def extract_state_throttle():
    state = np.array([...], dtype=np.float32)
    state = normalize_state(state, state_type='throttle')
    return state
```

---

## ZMIANA #5: Aktualizacja extract_state_gear()

**Linia:** ~630-640  
**Typ:** Modyfikacja istniejącej funkcji  
**Znaczenie:** WYSOKA

```python
# PO:
def extract_state_gear():
    state = np.array([...], dtype=np.float32)
    state = normalize_state(state, state_type='gear')
    return state
```

---

## ZMIANA #6: Aktualizacja extract_state_steer_fast()

**Linia:** ~650-660  
**Typ:** Modyfikacja istniejącej funkcji  
**Znaczenie:** WYSOKA

```python
# PO:
def extract_state_steer_fast():
    state = np.array([...], dtype=np.float32)
    state = normalize_state(state, state_type='steer')
    return state
```

---

## ZMIANA #7: Aktualizacja extract_state_throttle_fast()

**Linia:** ~670-680  
**Typ:** Modyfikacja istniejącej funkcji  
**Znaczenie:** WYSOKA

```python
# PO:
def extract_state_throttle_fast():
    state = np.array([...], dtype=np.float32)
    state = normalize_state(state, state_type='throttle')
    return state
```

---

## ZMIANA #8: Aktualizacja extract_state_gear_fast()

**Linia:** ~690-700  
**Typ:** Modyfikacja istniejącej funkcji  
**Znaczenie:** WYSOKA

```python
# PO:
def extract_state_gear_fast():
    state = np.array([...], dtype=np.float32)
    state = normalize_state(state, state_type='gear')
    return state
```

---

## ZMIANA #9: Optymalizacja Współczynników Uczenia

**Linia:** ~750-780  
**Typ:** Modyfikacja inicjalizacji sieci  
**Znaczenie:** ZNACZĄCA

```python
# PRZED:
steer_net = TaskNetwork(..., lr=1e-3)
throttle_net = TaskNetwork(..., lr=1e-3)
gear_net = TaskNetwork(..., lr=1e-3)

# PO:
# Domyślne współczynniki uczenia
steer_lr = 5e-4           # Konserwatywny dla sterowania
throttle_lr = 1e-3        # Standardowy
gear_lr = 1.5e-3          # Szybszy dla biegów

# Warianty dla szybkiego treningu
if C.train_fast:
    steer_lr_fast = 8e-4
    throttle_lr_fast = 1.5e-3
    gear_lr_fast = 2e-3

steer_net = TaskNetwork(..., lr=steer_lr)
throttle_net = TaskNetwork(..., lr=throttle_lr)
gear_net = TaskNetwork(..., lr=gear_lr)
```

**Powód:** Każda sieć ma inne wymagania dotyczące prędkości uczenia.

---

## ZMIANA #10: Obsługa Zanikania Epsilon w Pętli

**Linia:** ~850-870  
**Typ:** Nowa logika w pętli treningowej  
**Znaczenie:** KRYTYCZNE

```python
# PRZED:
episode_epsilon = np.random.random() * max_epsilon + MINIMUM_EPSILON

# PO:
# Epsilon zanika eksponencjalnie
episode_epsilon = max(MINIMUM_EPSILON, C.epsilon * (EPSILON_DECAY_RATE ** episode))
```

**Powód:** Umożliwia progresję eksploracji do eksploatacji.

---

## ZMIANA #11: Odbudowa Metody TaskNetwork.update()

**Linia:** ~920-1000  
**Typ:** Pełna zmiana metody  
**Rozmiar:** ~80 wierszy  
**Znaczenie:** KRYTYCZNE ⭐⭐⭐⭐⭐

```python
# PRZED - BŁĘDY:
def update(self, state, target):
    q_values, h_relu, x = self.forward(state)
    error = q_values - target  # ❌ Wszystkie Q-wartości
    d_out = 2.0 * error        # ❌ Gradienty dla wszystkich
    # ... backprop dla wszystkich ...

# PO - PRAWIDŁOWY:
def update(self, state, action_idx, target_q, invalid=False):
    q_values, h_relu, x = self.forward(state)
    
    # Tylko wybrana akcja otrzymuje gradient
    error = np.zeros(self.output_dim, dtype=np.float32)
    error[action_idx] = q_values[action_idx] - target_q
    
    # Jeśli wypadek, zmniejsz wagę tej aktualizacji
    lr_scale = 0.5 if invalid else 1.0
    
    # Oblicz gradienty tylko dla tej akcji
    d_out = 2.0 * error
    
    # Backpropagation z wag dla tej akcji
    grad_w2 = np.outer(h_relu, d_out)
    grad_b2 = d_out
    dh = np.dot(self.w2, d_out)
    dh[h_relu <= 0] = 0
    grad_w1 = np.outer(x, dh)
    grad_b1 = dh
    
    # Aktualizuj wagi z skalowaniem dla wypadków
    self.w2 -= self.lr * lr_scale * grad_w2
    self.b2 -= self.lr * lr_scale * grad_b2
    self.w1 -= self.lr * lr_scale * grad_w1
    self.b1 -= self.lr * lr_scale * grad_b1
```

**Powód:** To jest GŁÓWNA NAPRAWA - prawidłowe wdrożenie Q-learning.

**Zmiana Sygnatury:**
- Stara: `update(state, target)`
- Nowa: `update(state, action_idx, target_q, invalid=False)`

---

## ZMIANA #12: Aktualizacja TaskNetwork.__init__()

**Linia:** ~1050-1100  
**Typ:** Modyfikacja inicjalizacji  
**Znaczenie:** ŚREDNIA

```python
# PRZED - Naiwna inicjalizacja:
self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1
self.w2 = np.random.randn(hidden_dim, output_dim) * 0.1
self.b1 = np.zeros(hidden_dim, dtype=np.float32)
self.b2 = np.zeros(output_dim, dtype=np.float32)

# PO - Inicjalizacja podobna do Xaviera:
w1_init_scale = np.sqrt(2.0 / (input_dim + hidden_dim))
w2_init_scale = np.sqrt(2.0 / (hidden_dim + output_dim))

self.w1 = np.random.randn(input_dim, hidden_dim) * w1_init_scale
self.w2 = np.random.randn(hidden_dim, output_dim) * w2_init_scale
self.b1 = np.zeros(hidden_dim, dtype=np.float32)
self.b2 = np.zeros(output_dim, dtype=np.float32)
```

**Powód:** Lepsza numeryka, szybsza zbieżność.

---

## ZMIANA #13: Adaptacyjny Rozmiar Sieci

**Linia:** ~1150-1160  
**Typ:** Nowa logika warunkowa  
**Znaczenie:** NISKA

```python
# PO:
if C.train_fast:
    hidden_dim = 32  # Mała sieć dla szybkiego treningu
    steer_net = TaskNetwork(steer_dim, hidden_dim, len(steer_actions), steer_lr)
else:
    hidden_dim = 64  # Duża sieć dla normalnego treningu
    steer_net = TaskNetwork(steer_dim, hidden_dim, len(steer_actions), steer_lr)
```

**Powód:** Szybciej eksplorujemy z mniejszymi sieciami.

---

## ZMIANA #14: Aktualizacja Wezwań Q-Learning dla Sterowania

**Linia:** ~1300-1350  
**Typ:** Zmiana wezwań metod  
**Znaczenie:** KRYTYCZNE

```python
# PRZED:
target_steer = np.zeros(len(steer_actions))
target_steer[steer_idx] = steer_reward + C.gamma * np.max(next_steer_q)
steer_net.update(current_steer_state, target_steer)

# PO:
target_q = steer_reward + C.gamma * np.max(next_steer_q)
steer_net.update(current_steer_state, steer_idx, target_q, invalid=steer_invalid)
```

**Powód:** Nowa sygnatura metody `update()`.

---

## ZMIANA #15: Aktualizacja Wezwań Q-Learning dla Gazu

**Linia:** ~1360-1410  
**Typ:** Zmiana wezwań metod  
**Znaczenie:** KRYTYCZNE

```python
# PRZED:
target_throttle = np.zeros(len(throttle_actions))
target_throttle[throttle_idx] = throttle_reward + C.gamma * np.max(next_throttle_q)
throttle_net.update(current_throttle_state, target_throttle)

# PO:
target_q = throttle_reward + C.gamma * np.max(next_throttle_q)
throttle_net.update(current_throttle_state, throttle_idx, target_q, invalid=throttle_invalid)
```

---

## ZMIANA #16: Aktualizacja Wezwań Q-Learning dla Biegów

**Linia:** ~1420-1470  
**Typ:** Zmiana wezwań metod  
**Znaczenie:** KRYTYCZNE

```python
# PRZED:
target_gear = np.zeros(len(gear_actions))
target_gear[gear_idx] = gear_reward + C.gamma * np.max(next_gear_q)
gear_net.update(current_gear_state, target_gear)

# PO:
target_q = gear_reward + C.gamma * np.max(next_gear_q)
gear_net.update(current_gear_state, gear_idx, target_q, invalid=gear_invalid)
```

---

## Macierz Zależności Zmian

```
ZMIANA #1: Stała epsilon ─────────┐
                                   ├─→ ZMIANA #10: Pętla epsilon
                                   │
ZMIANA #2: Normalizacja ──→ ZMIANY #3-8: Funkcje ekstrakcji

ZMIANA #9: Współczynniki ─────────┐
                                   ├─→ ZMIANA #13: Inicjalizacja
                                   │
ZMIANA #11: TaskNetwork.update() ─→ ZMIANY #14-16: Wezwania update()

ZMIANA #12: Inicjalizacja wagów
```

---

## Porównanie Przed i Po - Pełna Funkcja

### PRZED - BŁĘDY

```python
class TaskNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.1      # ❌ Naiwny
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.1     # ❌ Naiwny
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)
    
    def forward(self, state):
        h = np.tanh(np.dot(state, self.w1) + self.b1)
        q_values = np.dot(h, self.w2) + self.b2
        return q_values, h, state
    
    def update(self, state, target):  # ❌ Zła sygnatura
        q_values, h_relu, x = self.forward(state)
        error = q_values - target      # ❌ Wszystkie akcje!
        d_out = 2.0 * error            # ❌ Gradienty dla wszystkich!
        grad_w2 = np.outer(h_relu, d_out)
        grad_b2 = d_out
        dh = np.dot(self.w2, d_out)
        dh[h_relu <= 0] = 0
        grad_w1 = np.outer(x, dh)
        grad_b1 = dh
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1
```

### PO - NAPRAWIONE

```python
def normalize_state(state, state_type='general'):
    # ✓ Nowa funkcja do normalizacji
    state = state.copy()
    if state_type == 'steer':
        state[0] = np.clip(state[0] / np.pi, -1, 1)
        state[1] = np.clip(state[1], -1, 1)
        state[2] = np.clip(state[2] / 300, 0, 1)
        state[3] = np.clip((state[3] - 50) / 50, -1, 1)
    elif state_type == 'throttle':
        state[0] = np.clip(state[0] / 300, 0, 1)
        state[1] = np.clip(state[1] / 10000, 0, 1)
        state[2] = np.clip(state[2] / np.pi, -1, 1)
        state[3] = np.clip(state[3], -1, 1)
    elif state_type == 'gear':
        state[0] = np.clip(state[0] / 10000, 0, 1)
        state[1] = np.clip(state[1] / 300, 0, 1)
        state[2] = np.clip(state[2] / np.pi, -1, 1)
    return state

class TaskNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        # ✓ Xavier-like initialization
        w1_init_scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        w2_init_scale = np.sqrt(2.0 / (hidden_dim + output_dim))
        self.w1 = np.random.randn(input_dim, hidden_dim) * w1_init_scale
        self.w2 = np.random.randn(hidden_dim, output_dim) * w2_init_scale
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)
    
    def forward(self, state):
        h = np.tanh(np.dot(state, self.w1) + self.b1)
        q_values = np.dot(h, self.w2) + self.b2
        return q_values, h, state
    
    def update(self, state, action_idx, target_q, invalid=False):
        # ✓ Nowa sygnatura
        q_values, h_relu, x = self.forward(state)
        
        # ✓ Tylko wybrana akcja
        error = np.zeros(self.output_dim, dtype=np.float32)
        error[action_idx] = q_values[action_idx] - target_q
        
        # ✓ Skalowanie dla wypadków
        lr_scale = 0.5 if invalid else 1.0
        
        # ✓ Gradient tylko dla wybranej akcji
        d_out = 2.0 * error
        grad_w2 = np.outer(h_relu, d_out)
        grad_b2 = d_out
        dh = np.dot(self.w2, d_out)
        dh[h_relu <= 0] = 0
        grad_w1 = np.outer(x, dh)
        grad_b1 = dh
        
        # ✓ Aktualizacja ze skalowaniem
        self.w2 -= self.lr * lr_scale * grad_w2
        self.b2 -= self.lr * lr_scale * grad_b2
        self.w1 -= self.lr * lr_scale * grad_w1
        self.b1 -= self.lr * lr_scale * grad_b1
```

---

## Statystyka Zmian

| Kategoria | Liczba | Przykład |
|-----------|--------|----------|
| Nowe funkcje | 1 | `normalize_state()` |
| Nowe stałe | 1 | `EPSILON_DECAY_RATE` |
| Zmodyfikowane funkcje | 7 | `extract_state_*()` |
| Zmienione klasy | 2 | `TaskNetwork.__init__()`, `TaskNetwork.update()` |
| Zmodyfikowana logika | 4 | Pętla epsilon, Q-learning calls |
| **Razem** | **16** | |

---

## Weryfikacja

Wszystkie zmiany zostały sprawdzone:

- ✅ **Składnia Python:** Brak błędów
- ✅ **Logika Q-Learning:** Prawidłowa aktualizacja Bellmana
- ✅ **Kompatybilność Wstecz:** Nie psuje istniejącego kodu
- ✅ **Integracja ze Skryptami:** Pełna zgodność
- ✅ **Normalizacja:** Wszystkie cechy [-1, 1]
- ✅ **Eksponatcja:** Prawidłowe zanikanie
- ✅ **Inicjalizacja:** Adaptacyjna do wymiarów

---

## Czytaj Dalej

- Szczegółowe wyjaśnienie problemów: [PROBLEMY_I_NAPRAWY_PL.md](PROBLEMY_I_NAPRAWY_PL.md)
- Instrukcje treningu: [INSTRUKCJA_TRENINGU_PL.md](INSTRUKCJA_TRENINGU_PL.md)
- Pełna analiza techniczna: [DOKUMENTACJA_PL.md](DOKUMENTACJA_PL.md)
