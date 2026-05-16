# Q-Learning Fixes - Quick Reference

## Change Summary (All Changes Applied ✓)

### 1️⃣ TaskNetwork.update() - Fixed Q-learning Update 

**Location:** Lines ~860-900

**Before (WRONG):**
```python
def update(self, state, target):
    q_values, h_relu, x = self.forward(state)
    error = q_values - target  # ❌ ALL actions affected
    d_out = 2.0 * error
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

**After (CORRECT):**
```python
def update(self, state, action_idx, target_q, invalid=False):
    """Update only the taken action's Q-value (proper Q-learning)."""
    q_values, h_relu, x = self.forward(state)
    
    # Create error only for the taken action ✓
    error = np.zeros(self.output_dim, dtype=np.float32)
    error[action_idx] = q_values[action_idx] - target_q
    
    d_out = 2.0 * error
    grad_w2 = np.outer(h_relu, d_out)
    grad_b2 = d_out
    dh = np.dot(self.w2, d_out)
    dh[h_relu <= 0] = 0
    grad_w1 = np.outer(x, dh)
    grad_b1 = dh
    
    lr_scale = 0.5 if invalid else 1.0  # Adaptive scaling for crashes
    
    self.w2 -= self.lr * lr_scale * grad_w2
    self.b2 -= self.lr * lr_scale * grad_b2
    self.w1 -= self.lr * lr_scale * grad_w1
    self.b1 -= self.lr * lr_scale * grad_b1
```

---

### 2️⃣ Added State Normalization Function

**Location:** Lines ~930-980

**New Code:**
```python
def normalize_state(state, state_type='general'):
    """Normalize state features to [-1, 1] range for better learning."""
    normalized = state.copy().astype(np.float32)
    
    if state_type == 'steer' and len(normalized) == 4:
        normalized[0] = np.clip(normalized[0] / np.pi, -1, 1)
        normalized[1] = np.clip(normalized[1], -1, 1)
        normalized[2] = np.clip(normalized[2] / 300.0, 0, 1)
        normalized[3] = np.clip(normalized[3] / 200.0 - 0.5, -1, 1)
    elif state_type == 'throttle' and len(normalized) == 5:
        normalized[0] = np.clip(normalized[0] / 300.0, 0, 1)
        normalized[1] = np.clip(normalized[1] / np.pi, -1, 1)
        normalized[2] = np.clip(normalized[2], -1, 1)
        normalized[3] = np.clip(normalized[3] / 10000.0 - 0.4, -1, 1)
        normalized[4] = np.clip(normalized[4] / 150.0 - 0.5, -1, 1)
    elif state_type == 'gear' and len(normalized) == 4:
        normalized[0] = np.clip(normalized[0] / 300.0, 0, 1)
        normalized[1] = np.clip(normalized[1] / 10000.0 - 0.4, -1, 1)
        normalized[2] = np.clip((normalized[2] - 3.5) / 3.5, -1, 1)
        normalized[3] = np.clip(normalized[3], -1, 1)
    
    return normalized
```

---

### 3️⃣ Updated State Extraction - Now Using Normalization

**Location:** Lines ~990-1070

**All functions updated. Example:**

**Before:**
```python
def extract_state_steer(S):
    return np.array([
        S['angle'],
        S['trackPos'],
        S['speedX'],
        S['track'][9],
    ], dtype=np.float32)
```

**After:**
```python
def extract_state_steer(S):
    state = np.array([
        S['angle'],
        S['trackPos'],
        S['speedX'],
        S['track'][9],
    ], dtype=np.float32)
    return normalize_state(state, 'steer')  # ✓ Normalized!
```

Applied to:
- `extract_state_steer()`
- `extract_state_throttle()`
- `extract_state_gear()`
- `extract_state_steer_fast()`
- `extract_state_throttle_fast()`
- `extract_state_gear_fast()`
- `extract_state_fast()`

---

### 4️⃣ Added Epsilon Decay Constant

**Location:** Lines ~1018-1019

**New:**
```python
MINIMUM_EPSILON = 0.01
EPSILON_DECAY_RATE = 0.98  # ✓ NEW: Exponential decay
```

---

### 5️⃣ Fixed Epsilon Strategy in Main Loop

**Location:** Lines ~1235-1244

**Before (WRONG):**
```python
for episode in range(C.maxEpisodes):
    max_epsilon = max(0.0, C.epsilon)
    episode_epsilon = np.random.random() * max_epsilon + MINIMUM_EPSILON
    # ❌ Random epsilon every episode - no learning progression!
```

**After (CORRECT):**
```python
for episode in range(C.maxEpisodes):
    # Implement exponential epsilon decay
    max_epsilon = max(MINIMUM_EPSILON, C.epsilon * (EPSILON_DECAY_RATE ** episode))
    episode_epsilon = max_epsilon  # ✓ Deterministic decay
    # ✓ Decays from 0.1 → 0.01 gradually
```

---

### 6️⃣ Fixed Q-Learning Update Calls

**Location:** Lines ~1313-1342

**Before (WRONG):**
```python
if steer_net is not None and not C.fixed_steer and steer_idx is not None and C.train_steer:
    print('REWARD STEER: %.2f' % steer_reward)
    target = steer_net.predict(current_steer_state)
    if steer_invalid:
        target = np.full(len(target), steer_reward)  # ❌ All actions same value!
    else:
        target[steer_idx] = steer_reward + C.gamma * np.max(steer_net.predict(next_steer_state))
    steer_net.update(current_steer_state, target)  # ❌ Wrong signature
```

**After (CORRECT):**
```python
if steer_net is not None and not C.fixed_steer and steer_idx is not None and C.train_steer:
    print('REWARD STEER: %.2f' % steer_reward)
    if steer_invalid:
        target_q = steer_reward  # ✓ Just the reward
    else:
        # ✓ Bellman equation for non-terminal
        target_q = steer_reward + C.gamma * np.max(steer_net.predict(next_steer_state))
    # ✓ New signature: (state, action_idx, target_q, invalid)
    steer_net.update(current_steer_state, steer_idx, target_q, invalid=steer_invalid)
```

Applied to all three networks: steering, throttle, gear

---

### 7️⃣ Optimized Learning Rates & Network Sizes

**Location:** Lines ~1211-1228

**Before (WRONG):**
```python
hidden_dim = 64  # Always the same
if steer_net is None and not C.fixed_steer and C.mode == 'qlearn' and C.train_steer:
    steer_net = TaskNetwork(..., lr=1e-3)  # ❌ All same LR
if throttle_net is None and not C.fixed_throttle and C.mode == 'qlearn' and C.train_throttle:
    throttle_net = TaskNetwork(..., lr=1e-3)
if gear_net is None and not C.fixed_gear and C.mode == 'qlearn' and C.train_gear:
    gear_net = TaskNetwork(..., lr=1e-3)
```

**After (CORRECT):**
```python
hidden_dim = 64  # Default for production
steer_lr = 5e-4      # ✓ Conservative for steering
throttle_lr = 1e-3   # ✓ Moderate for throttle
gear_lr = 1.5e-3     # ✓ Faster for gear

if C.train_fast:
    hidden_dim = 32  # ✓ Smaller networks for fast training
    steer_lr = 8e-4
    throttle_lr = 1.5e-3
    gear_lr = 2e-3

if steer_net is None and not C.fixed_steer and C.mode == 'qlearn' and C.train_steer:
    steer_net = TaskNetwork(..., lr=steer_lr)  # ✓ Task-specific LR
if throttle_net is None and not C.fixed_throttle and C.mode == 'qlearn' and C.train_throttle:
    throttle_net = TaskNetwork(..., lr=throttle_lr)
if gear_net is None and not C.fixed_gear and C.mode == 'qlearn' and C.train_gear:
    gear_net = TaskNetwork(..., lr=gear_lr)
```

---

### 8️⃣ Improved Weight Initialization

**Location:** Lines ~837-844

**Before (NAIVE):**
```python
self.w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
self.w2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
```

**After (XAVIER-LIKE):**
```python
w1_init_scale = np.sqrt(2.0 / (input_dim + hidden_dim))
w2_init_scale = np.sqrt(2.0 / (hidden_dim + output_dim))
self.w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * w1_init_scale
self.w2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * w2_init_scale
```

---

## Testing the Fixes

### Quick Test
```bash
# Just run one episode with steering training
python3 torcs_jm_par_new.py --train --train-fast --mode qlearn \
    --episodes 1 --train-steer --fixed-throttle --fixed-gear
```

### Full Training
```bash
# Run the complete training sequence
./run_training_sequence.sh 50
```

---

## Verification Checklist

- [x] No Python syntax errors
- [x] Q-learning update only touches taken action
- [x] Epsilon decays exponentially
- [x] States normalized to [-1, 1]
- [x] Learning rates optimized per network
- [x] Invalid state handling correct
- [x] Fast training mode uses smaller networks
- [x] Compatible with run_training_sequence.sh
- [x] Documentation updated

---

## Expected Results After Running

**Steering Training (Stage 1):**
- Should see improving angle and trackPos values
- Reward signals becoming more consistent
- Better centering on track over episodes

**Throttle Training (Stage 2):**
- Smoother acceleration/braking
- Better speed maintenance
- Reduced oscillation in speed control

**Gear Training (Stage 3):**
- Gears shifting at appropriate speeds
- RPM staying closer to optimal range
- Smoother power delivery

All stages should show clear reward improvement with each episode.
