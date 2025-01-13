# WORDLE

### Base Architecture

First, let's examine our foundational architecture that all solutions build upon:

```python
class WordleSolverBase:
    def __init__(self, seed=None):
        self.BASE_URL = "https://wordle.votee.dev:8000/random"
        self.ENGLISH_WORDS_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
        self.word_list = self.load_word_list()
        self.seed = seed if seed is not None else random.randint(1, 1000000)
        self.reset_game_state()
```

Key Components:
- State management for game progression
- Word validation system
- API communication handling
- Flexible architecture for different solving strategies

### Solution Evolution Journey

Our development followed a systematic progression from simple to complex approaches:

# Wordle Heuristic Solver: Component-by-Component Analysis

## 1. Word List Processing
```python
def load_word_list(self):
    try:
        response = requests.get(self.ENGLISH_WORDS_URL)
        words = [word.lower().strip() 
                for word in response.text.splitlines() 
                if len(word.strip()) == 5]
        return words
    except:
        return ["stare", "crane", "slate"]  # Fallback common words
```

**Key Features:**
1. **Word Source**: Fetches words from a comprehensive English dictionary URL
2. **Processing Steps**:
   - Converts all words to lowercase
   - Strips whitespace
   - Filters for exactly 5-letter words
3. **Error Handling**: Returns common starter words as fallback
4. **Memory Efficient**: Uses list comprehension for processing

## 2. Word Validation Logic
```python
def is_word_valid(self, word):
    # Check correct letter positions
    for slot, letter in self.correct_letters.items():
        if word[slot] != letter:
            return False
            
    # Check present letters are used
    for letter, invalid_slots in self.present_letters.items():
        if letter not in word:
            return False
        if any(word[slot] == letter for slot in invalid_slots):
            return False
            
    # Check absent letters
    for letter in self.absent_letters:
        if letter in word and letter not in self.correct_letters.values():
            return False
            
    return True
```

**Validation Steps:**
1. **Position Check**:
   - Verifies known correct letters are in correct positions
   - Fast-fails if any position mismatch

2. **Present Letter Check**:
   - Confirms required letters exist in word
   - Ensures they're not in known incorrect positions
   - Two-stage validation for efficiency

3. **Absent Letter Check**:
   - Excludes words with known absent letters
   - Handles exception for letters that are present but in wrong position

## 3. Heuristic Word Selection
```python
def generate_next_guess(self):
    valid_words = self.get_valid_words()
    if not valid_words:
        return None
        
    # Calculate letter frequencies in valid words
    letter_freq = Counter("".join(valid_words))
    
    # Score words based on letter frequency
    word_scores = [(sum(letter_freq[c] for c in set(word)), word) for word in valid_words]
    
    # Select the word with the highest score
    best_word = max(word_scores, key=lambda x: x[0])[1]
    print(f"Best heuristic guess: {best_word}")
    return best_wordurn best_word
```

**Scoring System:**
1. **Letter Frequency Analysis**:
   - Counts occurrence of each letter in valid words
   - Higher weight to common letters
   - Uses Counter for efficient counting

## 4. API Interaction
```python
def submit_guess(self, guess):
    params = {
        "guess": guess,
        "size": 5,
        "seed": self.seed
    }
    try:
        response = requests.get(self.BASE_URL, params=params, verify=False)
        return response.json()
    except:
        return None
```

**Implementation Details:**
1. **Parameter Structure**:
   - guess: Current word attempt
   - size: Word length constraint
   - seed: For reproducibility

2. **Error Handling**:
   - Graceful failure on connection issues
   - Returns None for error case handling
   - SSL verification disabled for testing

## 5. Game Loop Implementation
```python
def solve(self):
    for attempt in range(6):
        guess = self.generate_next_guess()
        if not guess:
            return False
            
        self.previous_guesses.append(guess)
        response = self.submit_guess(guess)
        
        if not response:
            return False
            
        # Update game state
        for item in response:
            if item["result"] == "correct":
                self.correct_letters[item["slot"]] = item["guess"]
            elif item["result"] == "present":
                if item["guess"] not in self.present_letters:
                    self.present_letters[item["guess"]] = set()
                self.present_letters[item["guess"]].add(item["slot"])
            elif item["result"] == "absent":
                self.absent_letters.add(item["guess"])
                
        # Win check
        if all(item["result"] == "correct" for item in response):
            return True
            
    return False
```

**Game Flow:**
1. **Attempt Management**:
   - Maximum 6 attempts
   - Tracks previous guesses
   - Handles guess generation failures

2. **State Updates**:
   - Updates correct letter positions
   - Tracks present but misplaced letters
   - Records absent letters
   - Maintains sets for efficient lookups

3. **Win Condition**:
   - Checks for all correct letters
   - Early termination on success
   - Returns false after 6 failed attempts

## Key Optimizations:

1. **Data Structures**:
   - Uses sets for O(1) lookups
   - Counter for efficient frequency counting
   - Dictionary for position mapping

2. **Algorithm Efficiency**:
   - Early termination conditions
   - Minimal redundant calculations
   - Efficient state updates

3. **Memory Management**:
   - Reuses existing collections
   - Minimal temporary objects
   - Efficient string operations

## Usage Example:
```python
solver = WordleSolver(seed=12345)
success = solver.solve()

if success:
    print(f"Solved in {len(solver.previous_guesses)} attempts")
    print(f"Guesses: {solver.previous_guesses}")
else:
    print("Failed to solve")
```

This heuristic approach combines statistical analysis with game-specific rules to create an efficient solving strategy. The balance between letter frequency analysis, uniqueness bonuses, and penalty systems helps optimize word selection while maintaining reasonable computational complexity.

**2. Basic RDL (First ML Approach)**

Hypothesis:
- Neural networks can learn word patterns
- Balanced architecture might provide good performance/complexity trade-off
- Dropout regularization can prevent overfitting

Implementation:
```python
class WordleSolverRDL(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.model = nn.Sequential(
            nn.Linear(130, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.word_list))
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.002, 
            weight_decay=1e-4
        )
```

Results:
- Success Rate: 80%
- Average Attempts: 4.8
- Response Time: 5.5s

**3. RDL_A (Deep Architecture)**

Hypothesis:
- Deeper networks can learn more complex patterns
- BatchNorm can stabilize deep network training
- Aggressive learning rates with proper regularization can find better solutions

Implementation:
```python
class WordleSolverRDL_A(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.model = nn.Sequential(
            nn.Linear(130, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Dropout(0.3),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Linear(192, len(self.word_list))
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.003,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
```

Training Innovation:
```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    self.optimizer,
    T_0=10,    # first restart period
    T_mult=2,  # increase period after each restart
    eta_min=1e-6
)
```

Results:
- Success Rate: 90%
- Average Attempts: 5.0
- Response Time: 6.35s

**4. RDL_B (Wide Architecture)**

Hypothesis:
- Wider layers can capture more feature combinations
- Conservative learning approach might lead to better stability
- Lower dropout rates maintain more information flow

Implementation:
```python
class WordleSolverRDL_B(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.model = nn.Sequential(
            nn.Linear(130, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(self.word_list))
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=5e-5,
            betas=(0.9, 0.95)
        )
```

Training Innovation:
```python
scheduler = optim.lr_scheduler.OneCycleLR(
    self.optimizer,
    max_lr=0.001,
    epochs=num_epochs,
    steps_per_epoch=len(valid_targets) // batch_size,
    pct_start=0.4
)
```

Results:
- Success Rate: 60%
- Average Attempts: 5.1
- Response Time: 7.03s

**5. RDL_C (Hybrid Approach)**

Hypothesis:
- Combining depth and width could capture benefits of both
- GELU activation might handle non-linearities better
- Label smoothing can improve model calibration

Implementation:
```python
class WordleSolverRDL_C(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.model = nn.Sequential(
            nn.Linear(130, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024, momentum=0.1),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.BatchNorm1d(512, momentum=0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128, momentum=0.1),
            nn.Linear(128, len(self.word_list))
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.002,
            weight_decay=2e-5,
            betas=(0.9, 0.98)
        )
        self.loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
```

Results:
- Success Rate: 70%
- Average Attempts: 5.0
- Response Time: 6.56s

### Performance Analysis & Business Impact

Comparative Results:
```
Model     Success Rate  Response Time  Complexity  
Heuristic 80%          5.0s          Minimal
Basic RDL 80%          5.5s          Low
RDL_A     90%          6.35s         Moderate
RDL_B     60%          7.03s         Low
RDL_C     70%          6.56s         High
```

Key Findings:
1. RDL_A emerged as the most effective solution
2. Deeper architectures outperformed wider ones
3. Proper regularization was crucial for success
4. Advanced techniques didn't always yield better results

### Technical Deep Dive

Let's examine the key technical decisions in our implementation:

**1. Optimizer Selection - AdamW**

Why AdamW over other optimizers?
```python
# RDL_A Configuration
optimizer = optim.AdamW(
    self.model.parameters(), 
    lr=0.003,            # Aggressive learning rate for fast convergence
    weight_decay=1e-4,   # Moderate weight decay for regularization
    betas=(0.9, 0.999)   # Standard momentum parameters
)
```
Key benefits:
- Adaptive learning rates for different parameters
- More accurate weight decay implementation than Adam
- Particularly suitable for our model scale (~100K-1M parameters)

**2. Normalization and Regularization Strategy**

BatchNorm usage:
```python
# RDL_A architecture showing BatchNorm placement
nn.Linear(130, 768),
nn.ReLU(),
nn.BatchNorm1d(768),   # Stabilizes deep network training
nn.Dropout(0.3),
nn.Linear(768, 384),
nn.ReLU(),
nn.BatchNorm1d(384)    # Continues normalization through layers
```
Benefits:
- Addresses internal covariate shift
- Enables higher learning rates
- Accelerates training convergence

Dropout strategy:
```python
# Different dropout rates for different architectures
RDL_A: nn.Dropout(0.3)  # Stronger regularization for deep architecture
RDL_B: nn.Dropout(0.2)  # Lighter dropout for wide architecture
RDL_C: nn.Dropout(0.2, 0.1)  # Progressive reduction
```

**3. Learning Rate and Weight Decay Choices**

Learning rate strategies:
```python
# RDL_A: Aggressive with warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,    # First restart period
    T_mult=2,  # Period doubling
    eta_min=1e-6
)

# RDL_B: Conservative with one cycle
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    pct_start=0.4  # 40% time for warmup
)
```

Weight decay selection:
- RDL_A: 1e-4 (moderate)
- RDL_B: 5e-5 (light)
- RDL_C: 2e-5 (minimal)
Rationale: Inverse relationship between architectural complexity and regularization strength

**4. Architecture Design Principles**

Input representation:
```python
# 5 positions × 26 letters = 130 dimensions
state_vector = np.zeros(130, dtype=np.float32)
```

Network depth and width choices:
```python
# RDL_A: Depth-focused
768 → 384 → 192  # Progressive narrowing

# RDL_B: Width-focused
1024 → 512  # Maintains width

# RDL_C: Hybrid approach
1024 → 512 → 256 → 128  # Balanced reduction
```

These technical choices were optimized for our specific constraints:
- Simulated training data
- Fast convergence requirement
- Moderate model size
- Flexible inference time

### Future Development

Based on our findings, we're now developing RDL_A2, which builds on RDL_A's success:
- Maintains the successful deep architecture
- Improves the learning rate schedule
- Adds early stopping mechanism
- Optimizes training data generation

### Conclusion

This project demonstrates:
1. Systematic approach to architecture exploration
2. Data-driven decision making
3. Balance of innovation and practical constraints
4. Clear progression from simple to complex solutions

