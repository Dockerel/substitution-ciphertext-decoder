import string, math, nltk
import pandas as pd
from random import shuffle, randint, uniform, sample


class CiphertextDecoder:
    def __init__(self, filename, decode_logging=False):
        self.alphabets = string.ascii_uppercase
        self.filename = filename
        self.ciphertext = ""
        self.original_case_info = []

        self.bigram_frequency = None
        self.trigram_frequency = None
        self.quadgram_frequency = None
        self.bigram_weight = 0.15
        self.trigram_weight = 0.35
        self.quadgram_weight = 0.5

        self.population_count = 500
        self.generation_count = 1000
        self.base_elite_percentage = 0.15
        self.elite_percentage = self.base_elite_percentage
        self.elite_population_count = round(
            self.population_count * self.elite_percentage
        )
        self.crossover_count = self.population_count - self.elite_population_count
        self.tournament_count = 15
        self.tournament_winner_probability = 0.75
        self.tournament_probabilities = [
            self.tournament_winner_probability
            * (1.0 - self.tournament_winner_probability) ** i
            for i in range(self.tournament_count)
        ]
        self.crossover_points_count = 3
        self.base_mutation_probability = 0.3
        self.mutation_probability = self.base_mutation_probability
        self.mutate_count = 3
        self.increment_rate = 0.0015
        self.decrement_rate = 0.0005
        self.terminate = 100

        self.decode_logging = decode_logging
        self.plaintext = ""

    def get_ngram_frequency(self, filename):
        file = pd.read_csv(filename)
        result = {}
        for _, row in file.iterrows():
            result[row["n-gram"]] = row["frequency"]
        return result

    def make_population(self):
        population = []
        for _ in range(self.population_count):
            key = list(self.alphabets)
            shuffle(key)
            population.append("".join(key))
        return population

    def decrypt(self, key):
        key_map = {}
        for i in range(26):
            key_map[self.alphabets[i]] = key[i]

        decrypted_text = ""
        for c in self.ciphertext:
            if c not in self.alphabets:
                decrypted_text += c
                continue
            decrypted_text += key_map[c]
        return decrypted_text

    def make_ngrams(self, text, n):
        ret = []
        for i in range(len(text) - n + 1):
            temp_ngram = text[i : i + n]
            if temp_ngram.isalpha():
                ret.append(temp_ngram)
        return ret

    def calculate_fitness(self, text):
        fitness = 0

        if self.bigram_weight > 0:
            bigram_fitness = 0
            ngrams = self.make_ngrams(text, 2)
            for ngram in ngrams:
                if ngram in self.bigram_frequency:
                    bigram_fitness += math.log2(self.bigram_frequency[ngram])
            fitness += bigram_fitness * self.bigram_weight

        if self.trigram_weight > 0:
            trigram_fitness = 0
            ngrams = self.make_ngrams(text, 3)
            for ngram in ngrams:
                if ngram in self.trigram_frequency:
                    trigram_fitness += math.log2(self.trigram_frequency[ngram])
            fitness += trigram_fitness * self.trigram_weight

        if self.quadgram_weight > 0:
            quadgram_fitness = 0
            ngrams = self.make_ngrams(text, 4)
            for ngram in ngrams:
                if ngram in self.quadgram_frequency:
                    quadgram_fitness += math.log2(self.quadgram_frequency[ngram])
            fitness += quadgram_fitness * self.quadgram_weight

        return fitness

    def evaluation(self, population):
        fitness = []

        for key in population:
            decrypted_text = self.decrypt(key)
            key_fitness = self.calculate_fitness(decrypted_text)
            fitness.append(key_fitness)

        return fitness

    def select_elite(self, population, fitness):
        population_fitness = {}
        for i in range(self.population_count):
            population_fitness[population[i]] = fitness[i]

        population_fitness = dict(
            sorted(population_fitness.items(), key=lambda x: x[1], reverse=True)
        )

        elite_population = list(population_fitness.keys())[
            : self.elite_population_count
        ]
        return elite_population

    def tournament_selection(self, population, fitness):
        population_copy = population.copy()
        parent = []

        for _ in range(2):
            tournament_population = {}
            for _ in range(self.tournament_count):
                r = randint(0, len(population_copy) - 1)
                tournament_population[population_copy[r]] = fitness[r]
                population_copy.pop(r)
            sorted_tournament_population = dict(
                sorted(tournament_population.items(), key=lambda x: x[1], reverse=True)
            )
            tournament_keys = list(sorted_tournament_population.keys())

            while 1:
                index = randint(0, self.tournament_count - 1)
                probability = self.tournament_probabilities[index]
                r = uniform(0, self.tournament_winner_probability)
                if probability > r:
                    parent.append(tournament_keys[index])
                    break

        return parent

    def merge_keys(self, parent1, parent2):
        offspring = [None] * 26
        random_indexes = sample(range(0, 26), self.crossover_points_count)
        for index in random_indexes:
            offspring[index] = parent1[index]

        curr = 0
        for i in range(0, 26):
            if offspring[i] != None:
                continue
            while 1:
                if parent2[curr] not in offspring:
                    offspring[i] = parent2[curr]
                    curr += 1
                    break
                curr += 1

        return "".join(offspring)

    def mutate_key(self, key):
        key = list(key)
        for _ in range(self.mutate_count):
            n1 = randint(0, 26 - 1)
            n2 = randint(0, 26 - 1)
            key[n1], key[n2] = key[n2], key[n1]
        return "".join(key)

    def mutation(self, population):
        for i in range(self.crossover_count):
            r = uniform(0, 1)
            if r <= self.mutation_probability:
                population[i] = self.mutate_key(population[i])
        return population

    def make_crossover(self, population, fitness):
        crossover_population = []
        while len(crossover_population) < self.crossover_count:
            parent1, parent2 = self.tournament_selection(population, fitness)

            offspring1 = self.merge_keys(parent1, parent2)
            offspring2 = self.merge_keys(parent2, parent1)

            crossover_population.extend([offspring1, offspring2])
        # population 갯수 맞춰주기
        if len(crossover_population) > self.crossover_count:
            crossover_population.pop()
        crossover_population = self.mutation(crossover_population)
        return crossover_population

    def convert_to_plaintext(self, fitness, population):
        try:
            max_fitness = max(fitness)

            index = fitness.index(max_fitness)
            key = population[index]
        except Exception:
            print(f"error at {index}")
        decrypted_text = self.decrypt(key)

        plaintext = ""
        for i in range(len(decrypted_text)):
            if self.original_case_info[i]:
                plaintext += decrypted_text[i].lower()
                continue
            plaintext += decrypted_text[i]

        self.plaintext = plaintext

    def change_parameter(self, stuck_counter):
        self.mutation_probability = self.base_mutation_probability + (
            self.increment_rate * stuck_counter
        )
        self.elite_percentage = self.base_elite_percentage - (
            self.decrement_rate * stuck_counter
        )
        self.elite_population_count = round(
            self.population_count * self.elite_percentage
        )
        self.crossover_count = self.population_count - self.elite_population_count

    def print_result(self, population, fitness, title, generation, stuck_counter):
        max_fitness = max(fitness)
        average_fitness = sum(fitness) / len(fitness)

        index = fitness.index(max_fitness)
        key = population[index]

        print(f"[{title} {generation} | stuck_counter: {stuck_counter}]")
        print(f"Average Fitness: {average_fitness}")
        print(f"Max Fitness: {max_fitness}")
        print(f"Key: {key}")
        print(f"Decrypted Text:\n{self.plaintext}\n")
        print("-" * 20 + "\n")

    def print_alphabet_frequency(self):
        frequency = nltk.FreqDist(self.make_ngrams(self.ciphertext, 1))
        sortedFrequency = dict(
            sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        )
        print("[Frequency of characters in the ciphertext ]")
        for k, v in sortedFrequency.items():
            print(f"- '{k}': {v}")
        print()

    def decode(self):
        self.ciphertext = open(self.filename, "r").read()
        self.original_case_info = [c.isalpha() and c.islower() for c in self.ciphertext]
        self.ciphertext = self.ciphertext.upper()

        filename2 = "data/2-ngramFrequency.csv"
        self.bigram_frequency = self.get_ngram_frequency(filename2)
        filename3 = "data/3-ngramFrequency.csv"
        self.trigram_frequency = self.get_ngram_frequency(filename3)
        filename4 = "data/4-ngramFrequency.csv"
        self.quadgram_frequency = self.get_ngram_frequency(filename4)

        population = self.make_population()

        max_fitness = -1
        stuck_counter = 0
        for generation in range(self.generation_count):
            # evaluation
            fitness = self.evaluation(population)

            self.convert_to_plaintext(fitness, population)
            if self.decode_logging:
                self.print_result(
                    population, fitness, "Generation", generation, stuck_counter
                )

            if max_fitness == max(fitness):
                stuck_counter += 1
            else:
                max_fitness = max(fitness)
                stuck_counter = 0
            if stuck_counter >= self.terminate:
                break

            self.change_parameter(stuck_counter)

            # elite population
            elite_population = self.select_elite(population, fitness)
            # crossover population
            crossover_population = self.make_crossover(population, fitness)
            population = elite_population + crossover_population

        if self.decode_logging:
            self.print_alphabet_frequency()
            self.print_result(
                population,
                fitness,
                "Decrypted result",
                f"| Generation: {generation}",
                stuck_counter,
            )

        return self.plaintext
