# ! apt install tesseract-ocr
# ! apt install libtesseract-dev
from PIL import Image
# !pip install pytesseract
import pytesseract
from pytesseract import image_to_string
# CUDA_LAUNCH_BLOCKING = "1"
import torch
# !pip install pytorch_pretrained_bert
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import re
import nltk
from min_edit_dist import find_word
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
##import error_correction_mod
# !apt install libenchant
# !pip install pyenchant
# !pip install --no-binary pyenchant
# !apt-list --installed | grep enchant
# !apt update
# !apt install enchant --fix-missing
from enchant.checker import SpellChecker
##!pip install textblob
##from textblob import TextBlob
##!pip install pyspellchecker
##from spellchecker import SpellChecker
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = (
    r'/usr/bin/tesseract'
)
filename = 'sample.png'
text = str(pytesseract.image_to_string(Image.open(filename)))
# text = "I amm a very generous perso. I donatd a fruit to a man toodayy."
print ("Original sentence: ","\n",text)

# cleanup text
rep = { '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ', 
        '"': ' " ', '"': ' " ', ',':' , ', '.':' . ', '!':' ! ', 
        '?':' ? ', "n't": " not" , "'ll": " will", '*':' * ', 
        '(': ' ( ', ')': ' ) ', "s'": "s '"}
rep = dict((re.escape(k), v) for k, v in rep.items()) 
pattern = re.compile("|".join(rep.keys()))
text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

print("\nMinimum edit distance result: ")
for word in text.split():
    pred = find_word(word)
    text = text.replace(word,pred)
print(text)
text_original = str(text)
def get_personslist(text):
    personslist=[]
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                personslist.insert(0, (chunk.leaves()[0][0]))
    return list(set(personslist))
personslist = get_personslist(text)
ignorewords = personslist + ["!", ",", ".", "\"", "?", "(", ")", "*", "'"]
# using SpellChecker identify incorrect words
d = SpellChecker("en_US")
words = text.split()

incorrectwords = [w for w in words if not d.check(w) and w not in ignorewords]
# using SpellChecker get suggested replacements
suggestedwords = [d.suggest(w) for w in incorrectwords]
# replace incorrect words with [MASK]

for w in incorrectwords:
    text = text.replace(w, '[MASK]')
    text_original = text_original.replace(w, '[MASK]')
    
print("\nMasked text","\n",text)

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Load, train and predict using pre-trained model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']
# Create the segments tensors
segs = [i for i, e in enumerate(tokenized_text) if e == "."]
segments_ids=[]
prev=-1
for k, s in enumerate(segs):
    segments_ids = segments_ids + [k] * (s-prev)
    prev=s
segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))
segments_tensors = torch.tensor([segments_ids])
# prepare Torch inputs 
tokens_tensor = torch.tensor([indexed_tokens])
# Load pre-trained model
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# Putting everything on CUDA
# tokens_tensor = tokens_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')
# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)
predictions

#Predict words for mask using BERT; 
#refine prediction by matching with proposals from SpellChecker
def predict_word(text_original, predictions, maskids):
    pred_words=[]
    for i in range(len(MASKIDS)):
        preds = torch.topk(predictions[0, MASKIDS[i]], k=50) 
        indices = preds.indices.tolist()
        list1 = tokenizer.convert_ids_to_tokens(indices)
        list2 = suggestedwords[i]
        simmax=0
        predicted_token=''
        for word1 in list1:
            for word2 in list2:
                s = SequenceMatcher(None, word1, word2).ratio()
                if s is not None and s > simmax:
                    simmax = s
                    predicted_token = word1
        text_original = text_original.replace('[MASK]', predicted_token, 1)
    return text_original
text_original = predict_word(text_original, predictions, MASKIDS)
print ("\nResult","\n",text_original)