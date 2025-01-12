import time
import pandas as pd
import matplotlib.pyplot as plt
import requests
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import random
import urllib3
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import torch.nn.functional as F

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Base Wordle Solver Class (unchanged)
class WordleSolverBase:
    def __init__(self, seed=None):
        self.BASE_URL = "https://wordle.votee.dev:8000/random"
        self.ENGLISH_WORDS_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
        self.word_list = self.load_word_list()
        self.seed = seed if seed is not None else random.randint(1, 1000000)
        self.reset_game_state()

    def reset_game_state(self):
        self.correct_letters = {}
        self.present_letters = {}
        self.absent_letters = set()
        self.min_letter_counts = Counter()
        self.max_letter_counts = Counter()
        self.previous_guesses = []

    def load_word_list(self):
        print("Loading word list...")
        try:
            response = requests.get(self.ENGLISH_WORDS_URL)
            response.raise_for_status()
            words = [word.lower().strip() for word in response.text.splitlines() if len(word.strip()) == 5]
            print(f"Loaded {len(words)} valid words.")
            return words
        except requests.exceptions.RequestException as e:
            print(f"Error loading word list: {e}")
            print("Using fallback word list.")
            return ["stare", "crane", "slate", "shine", "grime"]  # Fallback word list

    def submit_guess(self, guess):
        print(f"Submitting guess: {guess}")
        params = {"guess": guess, "size": 5, "seed": self.seed}
        try:
            response = requests.get(self.BASE_URL, params=params, verify=False)
            response.raise_for_status()
            response_data = response.json()
            if all(key in response_data[0] for key in ["guess", "result", "slot"]):
                print(f"API response: {response_data}")
                return response_data
            else:
                print("Invalid API response format.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error submitting guess: {e}")
            return None

    def is_word_valid(self, word):
        word_counter = Counter(word)
        for slot, letter in self.correct_letters.items():
            if word[slot] != letter:
                return False
        for letter, invalid_slots in self.present_letters.items():
            if letter not in word or any(word[slot] == letter for slot in invalid_slots):
                return False
        for letter, min_count in self.min_letter_counts.items():
            if word_counter[letter] < min_count:
                return False
        for letter in self.absent_letters:
            if letter in word and letter not in self.correct_letters.values() and letter not in self.present_letters:
                return False
        return True

    def get_valid_words(self):
        valid_words = [word for word in self.word_list if self.is_word_valid(word)]
        # print(f"Found {len(valid_words)} valid words.")
        return valid_words

    def solve(self):
        print(f"\nStarting game with seed: {self.seed}")
        for attempt in range(6):
            print(f"\nAttempt {attempt + 1}:")
            guess = self.generate_next_guess()
            if not guess:
                print("No valid words left to guess.")
                return False
            print(f"Generated guess: {guess}")
            self.previous_guesses.append(guess)  # Record the guess
            response = self.submit_guess(guess)
            if not response:
                return False
            for item in response:
                if item["result"] == "correct":
                    self.correct_letters[item["slot"]] = item["guess"]
                elif item["result"] == "present":
                    if item["guess"] not in self.present_letters:
                        self.present_letters[item["guess"]] = set()
                    self.present_letters[item["guess"]].add(item["slot"])
                elif item["result"] == "absent":
                    self.absent_letters.add(item["guess"])
            if all(item["result"] == "correct" for item in response):
                print(f"Success! The word is '{guess}'. Found in {attempt + 1} attempts.")
                return True
        print("Failed to find the word in 6 attempts.")
        return False

# Heuristic Wordle Solver
class WordleSolver(WordleSolverBase):
    def generate_next_guess(self):
        valid_words = self.get_valid_words()
        if not valid_words:
            return None
        letter_freq = Counter("".join(valid_words))
        word_scores = [(sum(letter_freq[c] for c in set(word)), word) for word in valid_words]
        best_word = max(word_scores, key=lambda x: x[0])[1]
        print(f"Best heuristic guess: {best_word}")
        return best_word

# over 80% accuracy
class WordleSolverRDL(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        ).to(self.device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=0.002, 
                                    weight_decay=1e-4) 
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_simulated_games()

    def encode_state(self, states=None):
        if states is None:
            state_vector = np.zeros(130, dtype=np.float32)
            for slot, letter in self.correct_letters.items():
                state_vector[ord(letter) - ord('a') + slot * 26] = 1
            for letter, invalid_slots in self.present_letters.items():
                base_idx = ord(letter) - ord('a')
                for slot in range(5):
                    if slot in invalid_slots:
                        state_vector[base_idx + slot * 26] = -0.5
                    else:
                        state_vector[base_idx + slot * 26] = 0.5
            for letter in self.absent_letters:
                for slot in range(5):
                    state_vector[ord(letter) - ord('a') + slot * 26] = -1
            return torch.from_numpy(state_vector).to(self.device)
        else:
            batch_vectors = np.zeros((len(states), 130), dtype=np.float32)
            for i, (correct_letters, present_letters, absent_letters) in enumerate(states):
                for slot, letter in correct_letters.items():
                    batch_vectors[i, ord(letter) - ord('a') + slot * 26] = 1
                for letter, invalid_slots in present_letters.items():
                    base_idx = ord(letter) - ord('a')
                    for slot in range(5):
                        if slot in invalid_slots:
                            batch_vectors[i, base_idx + slot * 26] = -0.5
                        else:
                            batch_vectors[i, base_idx + slot * 26] = 0.5
                for letter in absent_letters:
                    for slot in range(5):
                        batch_vectors[i, ord(letter) - ord('a') + slot * 26] = -1
            return torch.from_numpy(batch_vectors).to(self.device)
    
    def train_simulated_games(self):
        print("Training RDL model...")
        batch_size = 512
        num_epochs = 20
        valid_targets = random.sample(self.word_list, 5000) 

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,   
            T_mult=2 
        )

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = len(valid_targets) // batch_size
            
            for _ in range(batch_size):
                states = []
                target_indices = []
                target_words = random.sample(valid_targets, batch_size)

                for target_word in target_words:
                    correct_letters, present_letters, absent_letters = {}, {}, set()
                    num_known = random.randint(1, 3)  
                    positions = random.sample(range(5), num_known)
                    
                    for i in positions:
                        if random.random() < 0.6:  
                            correct_letters[i] = target_word[i]
                        else:
                            present_letters.setdefault(target_word[i], set()).add(i)

                    potential_absent = set(chr(i) for i in range(97, 123)) - set(target_word)
                    absent_letters.update(random.sample(list(potential_absent), 
                                                    random.randint(3, 6)))

                    states.append((correct_letters, present_letters, absent_letters))
                    target_indices.append(self.word_list.index(target_word))

                # convert states to tensors
                state_vectors = self.encode_state(states)
                target_indices = torch.tensor(target_indices, dtype=torch.long, device=self.device)

                # forward
                self.optimizer.zero_grad()
                word_scores = self.model(state_vectors)
                loss = self.loss_fn(word_scores, target_indices)

                # backward
                loss.backward()
                
                # add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / batch_size
            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

            # update learning rate
            scheduler.step(avg_loss)

            # stop training if the loss is below a threshold
            if avg_loss < 0.01:
                print("Reached target loss threshold. Stopping training.")
                break

    def generate_next_guess(self):
        valid_words = self.get_valid_words()
        if not valid_words:
            return None
        self.model.eval()
        with torch.no_grad():
            state_vector = self.encode_state()
            word_scores = self.model(state_vector.unsqueeze(0)).squeeze()
            valid_scores = [(word, word_scores[self.word_list.index(word)].item()) for word in valid_words]
            best_word = max(valid_scores, key=lambda x: x[1])[0]
            print(f"Best RDL guess: {best_word}")
            return best_word

    
# A. deeper network + more aggressive optimizer settings
class WordleSolverRDL_A(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # deeper network
        self.model = nn.Sequential(
            nn.Linear(130, 768),  # deepen the first layer
            nn.ReLU(),
            nn.BatchNorm1d(768),  # add BatchNorm
            nn.Dropout(0.3),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Dropout(0.3),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.Linear(192, len(self.word_list))
        ).to(self.device)
        
        # use a more aggressive optimizer setting
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.003,  # a higher learning rate
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_simulated_games()

    def encode_state(self, states=None):
        if states is None:
            state_vector = np.zeros(130, dtype=np.float32)
            for slot, letter in self.correct_letters.items():
                state_vector[ord(letter) - ord('a') + slot * 26] = 1
            for letter, invalid_slots in self.present_letters.items():
                base_idx = ord(letter) - ord('a')
                for slot in range(5):
                    if slot in invalid_slots:
                        state_vector[base_idx + slot * 26] = -0.5
                    else:
                        state_vector[base_idx + slot * 26] = 0.5
            for letter in self.absent_letters:
                for slot in range(5):
                    state_vector[ord(letter) - ord('a') + slot * 26] = -1
            return torch.from_numpy(state_vector).to(self.device)
        else:
            batch_vectors = np.zeros((len(states), 130), dtype=np.float32)
            for i, (correct_letters, present_letters, absent_letters) in enumerate(states):
                for slot, letter in correct_letters.items():
                    batch_vectors[i, ord(letter) - ord('a') + slot * 26] = 1
                for letter, invalid_slots in present_letters.items():
                    base_idx = ord(letter) - ord('a')
                    for slot in range(5):
                        if slot in invalid_slots:
                            batch_vectors[i, base_idx + slot * 26] = -0.5
                        else:
                            batch_vectors[i, base_idx + slot * 26] = 0.5
                for letter in absent_letters:
                    for slot in range(5):
                        batch_vectors[i, ord(letter) - ord('a') + slot * 26] = -1
            return torch.from_numpy(batch_vectors).to(self.device)

    def train_simulated_games(self):
        train_start_time = time.time()
        print("Training RDL model A...")
        batch_size = 512  # keep the batch size
        num_epochs = 80   # increase the number of epochs
        valid_targets = random.sample(self.word_list, 6000)  # increase the number of training samples

        # use a more aggressive learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,    # first restart period
            T_mult=2,  # increase the period after each restart
            eta_min=1e-6  # minimum learning rate
        )

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for _ in range(len(valid_targets) // batch_size):
                states = []
                target_indices = []
                target_words = random.sample(valid_targets, batch_size)

                for target_word in target_words:
                    correct_letters, present_letters, absent_letters = {}, {}, set()
                    num_known = random.randint(2, 4)
                    positions = random.sample(range(5), num_known)
                    
                    for i in positions:
                        if random.random() < 0.65:
                            correct_letters[i] = target_word[i]
                        else:
                            present_letters.setdefault(target_word[i], set()).add(i)

                    potential_absent = set(chr(i) for i in range(97, 123)) - set(target_word)
                    absent_letters.update(random.sample(list(potential_absent), 
                                                    random.randint(4, 8)))

                    states.append((correct_letters, present_letters, absent_letters))
                    target_indices.append(self.word_list.index(target_word))

                state_vectors = self.encode_state(states)
                target_indices = torch.tensor(target_indices, dtype=torch.long, device=self.device)

                self.optimizer.zero_grad()
                word_scores = self.model(state_vectors)
                loss = self.loss_fn(word_scores, target_indices)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / (len(valid_targets) // batch_size)
            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

            if avg_loss < 0.008:
                print("Reached target loss threshold. Stopping training.")
                break

        train_end_time = time.time()
        print(f"Training took {train_end_time - train_start_time:.2f} seconds")

    def generate_next_guess(self):
        valid_words = self.get_valid_words()
        if not valid_words:
            return None
        self.model.eval()
        with torch.no_grad():
            state_vector = self.encode_state()
            word_scores = self.model(state_vector.unsqueeze(0)).squeeze()
            valid_scores = [(word, word_scores[self.word_list.index(word)].item()) for word in valid_words]
            best_word = max(valid_scores, key=lambda x: x[1])[0]
            print(f"Best RDL guess: {best_word}")
            return best_word

# B. wider network + more conservative optimizer settings
class WordleSolverRDL_B(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # wider network
        self.model = nn.Sequential(
            nn.Linear(130, 1024),  # widen the first layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(self.word_list))
        ).to(self.device)
        
        # conservative optimizer settings
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.001,  # conservative learning rate
            weight_decay=5e-5,  # lower weight decay
            betas=(0.9, 0.95)  # more conservative beta values
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_simulated_games()

    def encode_state(self, states=None):
        if states is None:
            state_vector = np.zeros(130, dtype=np.float32)
            for slot, letter in self.correct_letters.items():
                state_vector[ord(letter) - ord('a') + slot * 26] = 1
            for letter, invalid_slots in self.present_letters.items():
                base_idx = ord(letter) - ord('a')
                for slot in range(5):
                    if slot in invalid_slots:
                        state_vector[base_idx + slot * 26] = -0.5
                    else:
                        state_vector[base_idx + slot * 26] = 0.5
            for letter in self.absent_letters:
                for slot in range(5):
                    state_vector[ord(letter) - ord('a') + slot * 26] = -1
            return torch.from_numpy(state_vector).to(self.device)
        else:
            batch_vectors = np.zeros((len(states), 130), dtype=np.float32)
            for i, (correct_letters, present_letters, absent_letters) in enumerate(states):
                for slot, letter in correct_letters.items():
                    batch_vectors[i, ord(letter) - ord('a') + slot * 26] = 1
                for letter, invalid_slots in present_letters.items():
                    base_idx = ord(letter) - ord('a')
                    for slot in range(5):
                        if slot in invalid_slots:
                            batch_vectors[i, base_idx + slot * 26] = -0.5
                        else:
                            batch_vectors[i, base_idx + slot * 26] = 0.5
                for letter in absent_letters:
                    for slot in range(5):
                        batch_vectors[i, ord(letter) - ord('a') + slot * 26] = -1
            return torch.from_numpy(batch_vectors).to(self.device)

    def train_simulated_games(self):
        train_start_time = time.time()
        print("Training RDL model B...")
        batch_size = 512
        num_epochs = 100  # increase the number of epochs
        valid_targets = random.sample(self.word_list, 5000)

        # use One Cycle learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.001,
            epochs=num_epochs,
            steps_per_epoch=len(valid_targets) // batch_size,
            pct_start=0.4,  # 40% of the time for warm-up
            div_factor=20.0,
            final_div_factor=1000.0
        )

        best_loss = float('inf')
        best_model_state = None
        patience = 8
        no_improve_count = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for _ in range(len(valid_targets) // batch_size):
                states = []
                target_indices = []
                target_words = random.sample(valid_targets, batch_size)

                for target_word in target_words:
                    correct_letters, present_letters, absent_letters = {}, {}, set()
                    num_known = random.randint(2, 3)  # lower the number of known letters
                    positions = random.sample(range(5), num_known)
                    
                    for i in positions:
                        if random.random() < 0.75:  # increase the probability of correct letters
                            correct_letters[i] = target_word[i]
                        else:
                            present_letters.setdefault(target_word[i], set()).add(i)

                    potential_absent = set(chr(i) for i in range(97, 123)) - set(target_word)
                    absent_letters.update(random.sample(list(potential_absent), 
                                                    random.randint(3, 6)))

                    states.append((correct_letters, present_letters, absent_letters))
                    target_indices.append(self.word_list.index(target_word))

                state_vectors = self.encode_state(states)
                target_indices = torch.tensor(target_indices, dtype=torch.long, device=self.device)

                self.optimizer.zero_grad()
                word_scores = self.model(state_vectors)
                loss = self.loss_fn(word_scores, target_indices)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # lower the gradient clipping threshold
                
                self.optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / (len(valid_targets) // batch_size)
            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = self.model.state_dict()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"No improvement for {patience} epochs. Loading best model and stopping.")
                self.model.load_state_dict(best_model_state)
                break

            if avg_loss < 0.005:
                print("Reached target loss threshold. Stopping training.")
                break

        train_end_time = time.time()
        print(f"Training took {train_end_time - train_start_time:.2f} seconds")        

    def generate_next_guess(self):
        valid_words = self.get_valid_words()
        if not valid_words:
            return None
        self.model.eval()
        with torch.no_grad():
            state_vector = self.encode_state()
            word_scores = self.model(state_vector.unsqueeze(0)).squeeze()
            valid_scores = [(word, word_scores[self.word_list.index(word)].item()) for word in valid_words]
            best_word = max(valid_scores, key=lambda x: x[1])[0]
            print(f"Best RDL guess: {best_word}")
            return best_word
        

# C. deeper and wider network + improved optimizer settings
class WordleSolverRDL_C(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # deeper and wider network
        self.model = nn.Sequential(
            nn.Linear(130, 1024),  # increase the width
            nn.GELU(),  # use GELU activation
            nn.BatchNorm1d(1024, momentum=0.1),  # change BatchNorm momentum
            nn.Dropout(0.2),  # lower dropout rate in order to increase capacity
            
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
        ).to(self.device)
        
        # use improved optimizer settings
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.002,  # lower learning rate to stabilize training
            weight_decay=2e-5,  # lower weight decay to prevent over-regularization
            betas=(0.9, 0.98)  # adjust beta values for a more stable optimizer
        )
        
        # use Label Smoothing Cross Entropy loss
        self.loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        self.train_simulated_games()

    def encode_state(self, states=None):
        if states is None:
            state_vector = np.zeros(130, dtype=np.float32)
            # enhance the signal strength for correct letters
            for slot, letter in self.correct_letters.items():
                state_vector[ord(letter) - ord('a') + slot * 26] = 1.0
            
            for letter, invalid_slots in self.present_letters.items():
                base_idx = ord(letter) - ord('a')
                for slot in range(5):
                    if slot in invalid_slots:
                        state_vector[base_idx + slot * 26] = -0.7  # increase the negative signal strength
                    else:
                        state_vector[base_idx + slot * 26] = 0.7   # increase the positive signal strength
            
            for letter in self.absent_letters:
                for slot in range(5):
                    state_vector[ord(letter) - ord('a') + slot * 26] = -1.0
            
            return torch.from_numpy(state_vector).to(self.device)
        else:
            batch_vectors = np.zeros((len(states), 130), dtype=np.float32)
            for i, (correct_letters, present_letters, absent_letters) in enumerate(states):
                for slot, letter in correct_letters.items():
                    batch_vectors[i, ord(letter) - ord('a') + slot * 26] = 1.0
                
                for letter, invalid_slots in present_letters.items():
                    base_idx = ord(letter) - ord('a')
                    for slot in range(5):
                        if slot in invalid_slots:
                            batch_vectors[i, base_idx + slot * 26] = -0.7
                        else:
                            batch_vectors[i, base_idx + slot * 26] = 0.7
                
                for letter in absent_letters:
                    for slot in range(5):
                        batch_vectors[i, ord(letter) - ord('a') + slot * 26] = -1.0
            
            return torch.from_numpy(batch_vectors).to(self.device)

    def train_simulated_games(self):
        train_start_time = time.time()
        print("Training RDL model C...")
        batch_size = 768  # increase the batch size to improve training efficiency
        num_epochs = 100  # increase the number of epochs to improve convergence
        valid_targets = random.sample(self.word_list, 8000)  # increase the number of training samples
        
        # use One Cycle learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.003,
            epochs=num_epochs,
            steps_per_epoch=len(valid_targets) // batch_size,
            pct_start=0.3,  # 30% of the time for warm-up
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        # include early stopping mechanism
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for _ in range(len(valid_targets) // batch_size):
                states = []
                target_indices = []
                target_words = random.sample(valid_targets, batch_size)
                
                # adjutted sample generation strategy
                for target_word in target_words:
                    correct_letters, present_letters, absent_letters = {}, {}, set()
                    # dynamically determine the number of known letters
                    known_info = random.choices(
                        [2, 3, 4],
                        weights=[0.3, 0.4, 0.3],  
                        k=1
                    )[0]
                    
                    positions = random.sample(range(5), known_info)
                    
                    # portion of correct letters
                    for i in positions:
                        if random.random() < 0.7:  # increase the probability of correct letters
                            correct_letters[i] = target_word[i]
                        else:
                            present_letters.setdefault(target_word[i], set()).add(i)
                    
                    # a more aggressive strategy for absent letters
                    remaining_letters = set(chr(i) for i in range(97, 123)) - set(target_word)
                    # choose absent letters based on common letter frequency
                    common_letters = set('etaoinshrdlcumwfgypbvkjxqz')
                    priority_absent = list(remaining_letters & common_letters)
                    if priority_absent:
                        absent_letters.update(random.sample(priority_absent, 
                                                        min(6, len(priority_absent))))
                    
                    states.append((correct_letters, present_letters, absent_letters))
                    target_indices.append(self.word_list.index(target_word))
                
                state_vectors = self.encode_state(states)
                target_indices = torch.tensor(target_indices, dtype=torch.long, device=self.device)
                
                self.optimizer.zero_grad()
                word_scores = self.model(state_vectors)
                loss = self.loss_fn(word_scores, target_indices)
                
                loss.backward()
                # use a more aggressive gradient clipping threshold
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                
                self.optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(valid_targets) // batch_size)
            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
            
            # early stopping mechanism
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
            
            if avg_loss < 0.006:  # adjust the target loss threshold
                print("Reached target loss threshold. Stopping training.")
                break

        train_end_time = time.time()
        print(f"Training took {train_end_time - train_start_time:.2f} seconds")

    def generate_next_guess(self):
        valid_words = self.get_valid_words()
        if not valid_words:
            return None
            
        self.model.eval()
        with torch.no_grad():
            state_vector = self.encode_state()
            word_scores = self.model(state_vector.unsqueeze(0)).squeeze()
            
            # add heuristic enhancements
            valid_scores = []
            for word in valid_words:
                base_score = word_scores[self.word_list.index(word)].item()
                
                # adjust the score based on the diversity of letters
                unique_letters = len(set(word))
                diversity_bonus = unique_letters / 5.0 * 0.2
                
                # adjust the score based on the frequency of common letters
                common_letters = sum(1 for c in word if c in 'etaoinshrdl')
                frequency_bonus = common_letters / 5.0 * 0.1
                
                final_score = base_score + diversity_bonus + frequency_bonus
                valid_scores.append((word, final_score))
            
            best_word = max(valid_scores, key=lambda x: x[1])[0]
            print(f"Best Enhanced guess: {best_word}")
            return best_word


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
class WordleSolverRDL_A2(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Keep the successful deeper architecture from A but add selective width
        self.model = nn.Sequential(
            # Input layer - slightly wider but not as extreme as B
            nn.Linear(130, 896),  # Moderate increase from 768
            nn.ReLU(),
            nn.BatchNorm1d(896),  # Keep BatchNorm that worked in A
            nn.Dropout(0.3),      # Keep dropout rate that worked in A
            
            # Hidden layers - maintain depth from A with moderate width
            nn.Linear(896, 448),  # Proportional reduction
            nn.ReLU(),
            nn.BatchNorm1d(448),
            nn.Dropout(0.3),
            
            nn.Linear(448, 224),
            nn.ReLU(),
            nn.BatchNorm1d(224),
            nn.Dropout(0.2),      # Slightly reduce dropout in later layers
            
            # Output layer
            nn.Linear(224, len(self.word_list))
        ).to(self.device)
        
        # Keep the aggressive optimizer settings that worked in A
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.003,             # Keep the successful learning rate
            weight_decay=1e-4,    # Keep the same weight decay
            betas=(0.9, 0.999)    # Standard beta values
        )
        
        # Keep standard cross entropy loss - no need for label smoothing yet
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.train_simulated_games()

    def train_simulated_games(self):
        train_start_time = time.time()
        print("Training RDL model A2...")
        batch_size = 512
        num_epochs = 60  # Moderate increase from A
        valid_targets = random.sample(self.word_list, 6000)  # Slight increase in training data
        
        # Keep the successful OneCycleLR scheduler with minor tweaks
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.003,
            epochs=num_epochs,
            steps_per_epoch=len(valid_targets) // batch_size,
            pct_start=0.3,        # Keep warm-up period
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        # Add early stopping with patience
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for _ in range(len(valid_targets) // batch_size):
                states = []
                target_indices = []
                target_words = random.sample(valid_targets, batch_size)
                
                # Keep the successful training data generation strategy from A
                for target_word in target_words:
                    correct_letters, present_letters, absent_letters = {}, {}, set()
                    num_known = random.randint(2, 4)  # Keep same range
                    positions = random.sample(range(5), num_known)
                    
                    for i in positions:
                        if random.random() < 0.65:  # Keep same probability
                            correct_letters[i] = target_word[i]
                        else:
                            present_letters.setdefault(target_word[i], set()).add(i)

                    potential_absent = set(chr(i) for i in range(97, 123)) - set(target_word)
                    absent_letters.update(random.sample(list(potential_absent), 
                                                    random.randint(4, 7)))  # Slightly adjusted range

                    states.append((correct_letters, present_letters, absent_letters))
                    target_indices.append(self.word_list.index(target_word))
                
                state_vectors = self.encode_state(states)
                target_indices = torch.tensor(target_indices, dtype=torch.long, device=self.device)
                
                self.optimizer.zero_grad()
                word_scores = self.model(state_vectors)
                loss = self.loss_fn(word_scores, target_indices)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Keep gradient clipping
                
                self.optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(valid_targets) // batch_size)
            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                self.model.load_state_dict(best_model_state)
                break
                
            if avg_loss < 0.007:  # Slightly more stringent threshold
                print("Reached target loss threshold. Stopping training.")
                break

        train_end_time = time.time()
        print(f"Training took {train_end_time - train_start_time:.2f} seconds")

    def encode_state(self, states=None):
        if states is None:
            state_vector = np.zeros(130, dtype=np.float32)
            for slot, letter in self.correct_letters.items():
                state_vector[ord(letter) - ord('a') + slot * 26] = 1
            for letter, invalid_slots in self.present_letters.items():
                base_idx = ord(letter) - ord('a')
                for slot in range(5):
                    if slot in invalid_slots:
                        state_vector[base_idx + slot * 26] = -0.5
                    else:
                        state_vector[base_idx + slot * 26] = 0.5
            for letter in self.absent_letters:
                for slot in range(5):
                    state_vector[ord(letter) - ord('a') + slot * 26] = -1
            return torch.from_numpy(state_vector).to(self.device)
        else:
            batch_vectors = np.zeros((len(states), 130), dtype=np.float32)
            for i, (correct_letters, present_letters, absent_letters) in enumerate(states):
                for slot, letter in correct_letters.items():
                    batch_vectors[i, ord(letter) - ord('a') + slot * 26] = 1
                for letter, invalid_slots in present_letters.items():
                    base_idx = ord(letter) - ord('a')
                    for slot in range(5):
                        if slot in invalid_slots:
                            batch_vectors[i, base_idx + slot * 26] = -0.5
                        else:
                            batch_vectors[i, base_idx + slot * 26] = 0.5
                for letter in absent_letters:
                    for slot in range(5):
                        batch_vectors[i, ord(letter) - ord('a') + slot * 26] = -1
            return torch.from_numpy(batch_vectors).to(self.device)


    def generate_next_guess(self):
        valid_words = self.get_valid_words()
        if not valid_words:
            return None
            
        self.model.eval()
        with torch.no_grad():
            state_vector = self.encode_state()
            word_scores = self.model(state_vector.unsqueeze(0)).squeeze()
            
            # Keep the successful word selection strategy from A
            # but add a small diversity bonus
            valid_scores = []
            for word in valid_words:
                base_score = word_scores[self.word_list.index(word)].item()
                # Small bonus for unique letters (5% max)
                unique_letters = len(set(word))
                diversity_bonus = unique_letters / 100.0
                
                final_score = base_score + diversity_bonus
                valid_scores.append((word, final_score))
            
            best_word = max(valid_scores, key=lambda x: x[1])[0]
            print(f"Best RDL_A2 guess: {best_word}")
            return best_word

# Benchmarking and Plotting
def benchmark_solver(solver_classes, num_trials=10):
    results = []
    for solver_class in solver_classes:
        total_training_time = time.time()
        solver = solver_class() 
        training_time = time.time() - total_training_time
        print(f"\n{solver_class.__name__} training time: {training_time:.2f} seconds")
        
        solving_times = []
        for trial in range(num_trials):
            print(f"\nStarting trial {trial + 1} of {num_trials}...")
            seed = random.randint(1, 1000000)
            solver = solver_class(seed=seed)
            start_time = time.time()
            success = solver.solve()
            solve_time = time.time() - start_time
            solving_times.append(solve_time)
            
            results.append({
                "solver": solver_class.__name__,
                "success": int(success),  # Convert boolean to int for averaging
                "attempts": len(solver.previous_guesses) if success else 6,
                "training_time": training_time,
                "solve_time": solve_time
            })
            
        avg_solve_time = sum(solving_times) / len(solving_times)
        print(f"{solver_class.__name__} average solving time: {avg_solve_time:.2f} seconds")
    
    return pd.DataFrame(results)

def plot_results(results, metrics=["attempts", "time"]):
    import matplotlib.pyplot as plt
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        results.boxplot(column=metric, by="solver")
        plt.title(f"Comparison of {metric.capitalize()}")
        plt.suptitle("")
        plt.xlabel("Solver")
        plt.ylabel(metric.capitalize())
        plt.savefig(f'{metric}_comparison.png')
        plt.close()  

if __name__ == "__main__":
    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    
    solver_classes = [WordleSolver, WordleSolverRDL_A2]
    results_df = benchmark_solver(solver_classes, num_trials=10)
    
    print("\nAggregated Results:")
    print(results_df.groupby("solver").mean().round(3))
    plot_results(results_df, metrics=["success", "attempts", "solve_time", "training_time"])