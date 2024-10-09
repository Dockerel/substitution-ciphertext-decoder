# ngram analyzer
# from ngram_analyzer import NgramAnalyzer
# ngramAnalyzer = NgramAnalyzer()
# for i in range(2, 5):
#     ngramAnalyzer.extractNgrams(i)

# parser
from ciphertext_decoder import CiphertextDecoder

ciphertextDecoder = CiphertextDecoder("ciphertext/ciphertext2.txt", True)
ciphertextDecoder.decode()
