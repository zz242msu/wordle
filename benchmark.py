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

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Base Wordle Solver Class
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


# ML-based Wordle Solver
class WordleSolverML(WordleSolverBase):
    def __init__(self, seed=None):
        super().__init__(seed)
        print("Initializing ML model...")
        self.vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 2))
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.train_model()

    def train_model(self):
        print("Training ML model...")
        X = self.vectorizer.fit_transform(self.word_list)
        letter_freq = Counter("".join(self.word_list))
        total_letters = sum(letter_freq.values())
        letter_prob = {letter: count / total_letters for letter, count in letter_freq.items()}
        y = np.array([sum(letter_prob[char] for char in word) for word in self.word_list])
        self.model.fit(X, y)
        print("ML model trained.")

    def generate_next_guess(self):
        valid_words = self.get_valid_words()
        if not valid_words:
            return None
        X = self.vectorizer.transform(valid_words)
        scores = self.model.predict(X)
        best_word = valid_words[np.argmax(scores)]
        print(f"Best ML guess: {best_word}")
        return best_word


# Benchmarking and Plotting
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
    solver_classes = [WordleSolver, WordleSolverML]
    results = benchmark_solver(solver_classes, num_trials=10)
    print("\nAggregated Results:")
    print(results.groupby("solver").mean())
    plot_results(results, metrics=["attempts", "time"])