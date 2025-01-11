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
        print(f"Found {len(valid_words)} valid words.")
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

# Heuristic Wordle Solver (unchanged)
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

# # ML-based Wordle Solver (unchanged)
# class WordleSolverML(WordleSolverBase):
#     def __init__(self, seed=None):
#         super().__init__(seed)
#         print("Initializing ML model...")
#         self.vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 2))
#         self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#         self.train_model()

#     def train_model(self):
#         print("Training ML model...")
#         X = self.vectorizer.fit_transform(self.word_list)
#         letter_freq = Counter("".join(self.word_list))
#         total_letters = sum(letter_freq.values())
#         letter_prob = {letter: count / total_letters for letter, count in letter_freq.items()}
#         y = np.array([sum(letter_prob[char] for char in word) for word in self.word_list])
#         self.model.fit(X, y)
#         print("ML model trained.")

#     def generate_next_guess(self):
#         valid_words = self.get_valid_words()
#         if not valid_words:
#             return None
#         X = self.vectorizer.transform(valid_words)
#         scores = self.model.predict(X)
#         best_word = valid_words[np.argmax(scores)]
#         print(f"Best ML guess: {best_word}")
#         return best_word

# # 可行方案 over 80% accuracy, 效果首次比启发式好
# class WordleSolverRDL(WordleSolverBase):
#     def __init__(self, seed=None):
#         super().__init__(seed)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.model = nn.Sequential(
#         #     nn.Linear(130, 256),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.2),
#         #     nn.Linear(256, 128),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.2),
#         #     nn.Linear(128, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, len(self.word_list))
#         # ).to(self.device)
#         # 模型架构改进
#         self.model = nn.Sequential(
#             nn.Linear(130, 512),  # 增大第一层
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),  # 增加中间层容量
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, len(self.word_list))
#         ).to(self.device)
#         # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
#         # 使用更强的优化器
#         self.optimizer = optim.AdamW(self.model.parameters(), 
#                                     lr=0.002,  # 略微提高初始学习率
#                                     weight_decay=1e-4)  # 增加正则化
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.train_simulated_games()

#     def encode_state(self, states=None):
#         if states is None:
#             state_vector = np.zeros(130, dtype=np.float32)
#             for slot, letter in self.correct_letters.items():
#                 state_vector[ord(letter) - ord('a') + slot * 26] = 1
#             for letter, invalid_slots in self.present_letters.items():
#                 base_idx = ord(letter) - ord('a')
#                 for slot in range(5):
#                     if slot in invalid_slots:
#                         state_vector[base_idx + slot * 26] = -0.5
#                     else:
#                         state_vector[base_idx + slot * 26] = 0.5
#             for letter in self.absent_letters:
#                 for slot in range(5):
#                     state_vector[ord(letter) - ord('a') + slot * 26] = -1
#             return torch.from_numpy(state_vector).to(self.device)
#         else:
#             batch_vectors = np.zeros((len(states), 130), dtype=np.float32)
#             for i, (correct_letters, present_letters, absent_letters) in enumerate(states):
#                 for slot, letter in correct_letters.items():
#                     batch_vectors[i, ord(letter) - ord('a') + slot * 26] = 1
#                 for letter, invalid_slots in present_letters.items():
#                     base_idx = ord(letter) - ord('a')
#                     for slot in range(5):
#                         if slot in invalid_slots:
#                             batch_vectors[i, base_idx + slot * 26] = -0.5
#                         else:
#                             batch_vectors[i, base_idx + slot * 26] = 0.5
#                 for letter in absent_letters:
#                     for slot in range(5):
#                         batch_vectors[i, ord(letter) - ord('a') + slot * 26] = -1
#             return torch.from_numpy(batch_vectors).to(self.device)

#     # w/o 学习率衰减
#     # def train_simulated_games(self):
#     #     print("Training RDL model...")
#     #     batch_size = 256
#     #     num_epochs = 20
#     #     valid_targets = random.sample(self.word_list, 5000)  # Use a smaller subset for training

#     #     for epoch in range(num_epochs):
#     #         self.model.train()
#     #         total_loss = 0
#     #         for _ in range(batch_size):
#     #             states = []
#     #             target_indices = []
#     #             target_words = random.sample(valid_targets, batch_size)

#     #             for target_word in target_words:
#     #                 correct_letters, present_letters, absent_letters = {}, {}, set()
#     #                 # Randomly simulate known information
#     #                 for i in range(5):
#     #                     if random.random() < 0.3:
#     #                         correct_letters[i] = target_word[i]
#     #                     elif random.random() < 0.3:
#     #                         present_letters.setdefault(target_word[i], set()).add(i)
#     #                 absent_letters.update({chr(random.randint(97, 122)) for _ in range(5)} - set(target_word))

#     #                 states.append((correct_letters, present_letters, absent_letters))
#     #                 target_indices.append(self.word_list.index(target_word))  # Store index of target word

#     #             # Encode states into feature vectors
#     #             state_vectors = self.encode_state(states)
#     #             target_indices = torch.tensor(target_indices, dtype=torch.long, device=self.device)  # Convert to tensor

#     #             # Forward pass
#     #             word_scores = self.model(state_vectors)

#     #             # Compute loss using target indices
#     #             loss = self.loss_fn(word_scores, target_indices)

#     #             # Backward pass
#     #             self.optimizer.zero_grad()
#     #             loss.backward()
#     #             self.optimizer.step()

#     #             total_loss += loss.item()

#     #         avg_loss = total_loss / batch_size
#     #         print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

#     #         # Early stopping
#     #         if avg_loss < 0.01:
#     #             print("Early stopping due to convergence.")
#     #             break

#     def train_simulated_games(self):
#         print("Training RDL model...")
#         batch_size = 512
#         # num_epochs = 20
#         num_epochs = 40 # 增加训练轮数
#         valid_targets = random.sample(self.word_list, 5000)  # 保持原有训练集大小

#         # # 添加学习率衰减策略
#         # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         #     self.optimizer,
#         #     mode='min',
#         #     # factor=0.5,
#         #     # patience=2,
#         #     factor=0.3,    # 改为0.3让学习率下降更温和
#         #     patience=3,    # 增加耐心值
#         #     verbose=True
#         # )

#         # 使用更激进的学习率调度
#         scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             self.optimizer,
#             T_0=5,    # 第一次重启的周期长度
#             T_mult=2  # 每次重启后周期长度翻倍
#         )

#         for epoch in range(num_epochs):
#             self.model.train()
#             total_loss = 0
#             num_batches = len(valid_targets) // batch_size
            
#             for _ in range(batch_size):
#                 states = []
#                 target_indices = []
#                 target_words = random.sample(valid_targets, batch_size)

#                 for target_word in target_words:
#                     correct_letters, present_letters, absent_letters = {}, {}, set()
#                     # 优化状态模拟
#                     num_known = random.randint(1, 3)  # 确保至少有一些已知信息
#                     positions = random.sample(range(5), num_known)
                    
#                     for i in positions:
#                         if random.random() < 0.6:  # 60%概率是正确位置
#                             correct_letters[i] = target_word[i]
#                         else:
#                             present_letters.setdefault(target_word[i], set()).add(i)

#                     # 添加合理数量的不存在字母
#                     potential_absent = set(chr(i) for i in range(97, 123)) - set(target_word)
#                     absent_letters.update(random.sample(list(potential_absent), 
#                                                     random.randint(3, 6)))

#                     states.append((correct_letters, present_letters, absent_letters))
#                     target_indices.append(self.word_list.index(target_word))

#                 # 转换状态为特征向量
#                 state_vectors = self.encode_state(states)
#                 target_indices = torch.tensor(target_indices, dtype=torch.long, device=self.device)

#                 # 前向传播
#                 self.optimizer.zero_grad()
#                 word_scores = self.model(state_vectors)
#                 loss = self.loss_fn(word_scores, target_indices)

#                 # 反向传播
#                 loss.backward()
                
#                 # 添加梯度裁剪
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
#                 self.optimizer.step()
#                 total_loss += loss.item()

#             avg_loss = total_loss / batch_size
#             print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

#             # 更新学习率
#             scheduler.step(avg_loss)

#             # 如果损失足够小则提前停止
#             if avg_loss < 0.01:
#                 print("Reached target loss threshold. Stopping training.")
#                 break

#     def generate_next_guess(self):
#         valid_words = self.get_valid_words()
#         if not valid_words:
#             return None
#         self.model.eval()
#         with torch.no_grad():
#             state_vector = self.encode_state()
#             word_scores = self.model(state_vector.unsqueeze(0)).squeeze()
#             valid_scores = [(word, word_scores[self.word_list.index(word)].item()) for word in valid_words]
#             best_word = max(valid_scores, key=lambda x: x[1])[0]
#             print(f"Best RDL guess: {best_word}")
#             return best_word

    
# 改进方案A：更深的网络 + 激进的学习策略
class WordleSolverRDL(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 更深的网络结构
        self.model = nn.Sequential(
            nn.Linear(130, 768),  # 显著增大第一层
            nn.ReLU(),
            nn.BatchNorm1d(768),  # 添加BatchNorm
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
        
        # 使用更激进的优化器设置
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.003,  # 更高的初始学习率
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_simulated_games()

    # 两个方案共用的编码方法
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
        print("Training RDL model A...")
        batch_size = 512  # 保持较大的batch size
        num_epochs = 80   # 显著增加训练轮数
        valid_targets = random.sample(self.word_list, 6000)  # 增加训练样本

        # 使用周期性学习率重启
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,    # 第一次重启的周期
            T_mult=2,  # 每次重启后周期翻倍
            eta_min=1e-6  # 最小学习率
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

# 改进方案B：宽而浅的网络 + 保守的学习策略
class WordleSolverRDL_B(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 宽而浅的网络
        self.model = nn.Sequential(
            nn.Linear(130, 1024),  # 非常宽的第一层
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(self.word_list))
        ).to(self.device)
        
        # 保守的优化器设置
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.001,  # 保守的学习率
            weight_decay=5e-5,  # 较小的权重衰减
            betas=(0.9, 0.95)  # 更保守的动量参数
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_simulated_games()

    def train_simulated_games(self):
        print("Training RDL model B...")
        batch_size = 512
        num_epochs = 100  # 使用更多epoch进行充分训练
        valid_targets = random.sample(self.word_list, 5000)

        # 使用更温和的学习率调度
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.001,
            epochs=num_epochs,
            steps_per_epoch=len(valid_targets) // batch_size,
            pct_start=0.4,  # 40%时间用于预热
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
                    num_known = random.randint(2, 3)  # 稍微减少已知信息
                    positions = random.sample(range(5), num_known)
                    
                    for i in positions:
                        if random.random() < 0.75:  # 提高正确位置的概率
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # 更保守的梯度裁剪
                
                self.optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / (len(valid_targets) // batch_size)
            print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

            # 保存最佳模型
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

# # RDL with ResidualBlock
# class WordleSolverRDL(WordleSolverBase):
#     def __init__(self, seed=None):
#         super().__init__(seed)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # 增强型网络架构
#         self.model = nn.Sequential(
#             nn.Linear(130, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             ResidualBlock(512),
#             ResidualBlock(512),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             ResidualBlock(256),
#             nn.Linear(256, len(self.word_list))
#         ).to(self.device)
        
#         # 使用 AdamW 优化器
#         self.optimizer = optim.AdamW(
#             self.model.parameters(),
#             lr=0.002,
#             weight_decay=1e-4,
#             betas=(0.9, 0.999)
#         )
        
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.train_simulated_games()

#     def encode_state(self, states=None):
#         if states is None:
#             state_vector = np.zeros(130, dtype=np.float32)
#             for slot, letter in self.correct_letters.items():
#                 state_vector[ord(letter) - ord('a') + slot * 26] = 1
#             for letter, invalid_slots in self.present_letters.items():
#                 base_idx = ord(letter) - ord('a')
#                 for slot in range(5):
#                     if slot in invalid_slots:
#                         state_vector[base_idx + slot * 26] = -0.5
#                     else:
#                         state_vector[base_idx + slot * 26] = 0.5
#             for letter in self.absent_letters:
#                 for slot in range(5):
#                     state_vector[ord(letter) - ord('a') + slot * 26] = -1
#             return torch.from_numpy(state_vector).to(self.device)
#         else:
#             batch_vectors = np.zeros((len(states), 130), dtype=np.float32)
#             for i, (correct_letters, present_letters, absent_letters) in enumerate(states):
#                 for slot, letter in correct_letters.items():
#                     batch_vectors[i, ord(letter) - ord('a') + slot * 26] = 1
#                 for letter, invalid_slots in present_letters.items():
#                     base_idx = ord(letter) - ord('a')
#                     for slot in range(5):
#                         if slot in invalid_slots:
#                             batch_vectors[i, base_idx + slot * 26] = -0.5
#                         else:
#                             batch_vectors[i, base_idx + slot * 26] = 0.5
#                 for letter in absent_letters:
#                     for slot in range(5):
#                         batch_vectors[i, ord(letter) - ord('a') + slot * 26] = -1
#             return torch.from_numpy(batch_vectors).to(self.device)

#     def train_simulated_games(self):
#         print("Training RDL model...")
#         batch_size = 512  # 增大batch size
#         num_epochs = 20
#         valid_targets = random.sample(self.word_list, 5000)

#         # 使用 CosineAnnealingWarmRestarts 调度器
#         scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             self.optimizer,
#             T_0=5,    # 第一次重启的周期长度
#             T_mult=2  # 每次重启后周期长度翻倍
#         )

#         for epoch in range(num_epochs):
#             self.model.train()
#             total_loss = 0
#             num_batches = len(valid_targets) // batch_size
            
#             for _ in range(batch_size):
#                 states = []
#                 target_indices = []
#                 target_words = random.sample(valid_targets, batch_size)

#                 for target_word in target_words:
#                     correct_letters, present_letters, absent_letters = {}, {}, set()
#                     num_known = random.randint(1, 3)
#                     positions = random.sample(range(5), num_known)
                    
#                     for i in positions:
#                         if random.random() < 0.6:
#                             correct_letters[i] = target_word[i]
#                         else:
#                             present_letters.setdefault(target_word[i], set()).add(i)

#                     potential_absent = set(chr(i) for i in range(97, 123)) - set(target_word)
#                     absent_letters.update(random.sample(list(potential_absent), 
#                                                      random.randint(3, 6)))

#                     states.append((correct_letters, present_letters, absent_letters))
#                     target_indices.append(self.word_list.index(target_word))

#                 state_vectors = self.encode_state(states)
#                 target_indices = torch.tensor(target_indices, dtype=torch.long, device=self.device)

#                 self.optimizer.zero_grad()
#                 word_scores = self.model(state_vectors)
#                 loss = self.loss_fn(word_scores, target_indices)
#                 loss.backward()
                
#                 # 更激进的梯度裁剪
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
#                 self.optimizer.step()
#                 scheduler.step()  # 每个batch都更新学习率
                
#                 total_loss += loss.item()

#             avg_loss = total_loss / batch_size
#             print(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

#     def generate_next_guess(self):
#         valid_words = self.get_valid_words()
#         if not valid_words:
#             return None
#         self.model.eval()
#         with torch.no_grad():
#             state_vector = self.encode_state()
#             word_scores = self.model(state_vector.unsqueeze(0)).squeeze()
#             valid_scores = [(word, word_scores[self.word_list.index(word)].item()) for word in valid_words]
#             best_word = max(valid_scores, key=lambda x: x[1])[0]
#             print(f"Best RDL guess: {best_word}")
#             return best_word


# Benchmarking and Plotting (unchanged)
def benchmark_solver(solver_classes, num_trials=10):
    results = []
    for trial in range(num_trials):
        print(f"\nStarting trial {trial + 1} of {num_trials}...")
        seed = random.randint(1, 1000000)  # Generate a seed for this trial
        print(f"Using seed: {seed}")
        for solver_class in solver_classes:
            print(f"\nTesting solver: {solver_class.__name__}")
            solver = solver_class(seed=seed)  # Use the same seed for all solvers
            start_time = time.time()
            success = solver.solve()
            end_time = time.time()
            results.append({
                "solver": solver_class.__name__,
                "success": success,
                "attempts": len(solver.previous_guesses) if success else 6,
                "time": end_time - start_time,
                "seed": seed  # Record the seed for debugging
            })
    return pd.DataFrame(results)

def plot_results(results, metrics=["attempts", "time"]):
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        results.boxplot(column=metric, by="solver")
        plt.title(f"Comparison of {metric.capitalize()}")
        plt.suptitle("")
        plt.xlabel("Solver")
        plt.ylabel(metric.capitalize())
        plt.show()

if __name__ == "__main__":
    # solver_classes = [WordleSolver, WordleSolverML, WordleSolverRDL]
    solver_classes = [WordleSolver, WordleSolverRDL]
    results = benchmark_solver(solver_classes, num_trials=10)
    print("\nAggregated Results:")
    print(results.groupby("solver").mean())
    plot_results(results, metrics=["attempts", "time"])
