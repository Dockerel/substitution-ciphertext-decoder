from nltk import FreqDist
import pandas as pd


class NgramAnalyzer:

    def __init__(self, n=0):
        self.n = n

    def inputParam(self):
        while 1:
            n = int(input("n-gram length: "))

            if n >= 1:
                self.n = n
                break
            print("n-gram length should be over 1")

    def makeNgrams(self, word):
        ret = []
        for i in range(0, len(word) - self.n + 1):
            tempNgram = "".join(word[i : i + self.n])
            if tempNgram.isalpha():
                ret.append(tempNgram.upper())
        return ret

    def extractNgrams(self, n=0):
        if n > 0:
            self.n = n

        file = open("TheAdventuresOfSherlockHolmes.txt", "r").read()

        words = file.split()

        extracted_ngrams = []
        for word in words:
            result = self.makeNgrams(word)
            extracted_ngrams.extend(result)
        frequency = FreqDist(extracted_ngrams)
        sortedFrequency = dict(
            sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        )

        data = [
            {"n-gram": ngram, "frequency": freq}
            for ngram, freq in sortedFrequency.items()
        ]

        df = pd.DataFrame(data)
        df.to_csv(f"data/{self.n}-ngramFrequency.csv", index=False)

        print(f"ngram({self.n}) data saved successfully.")
