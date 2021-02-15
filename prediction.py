import pickle
import time
import numpy as np
from collections import defaultdict
import re
import csv
from bert_serving.client import BertClient
import joblib
import pandas as pd
# from empath import Empath

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d", " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def text_preprocessing_1(data, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    vocab = defaultdict(float)
    status = []
    m = []
    sentences = re.split(r'[.]', data.strip())
    # print(sentences)
    try:
        sentences.remove('')
    except ValueError:
        pass
        # print("omg")
    sentences = [sent + "." for sent in sentences]
    last_sentences = []
    for i in range(len(sentences)):
        sents = re.split(r'[?]', sentences[i].strip())
        for s in sents:
            try:
                if len(s) == 0:
                    pass
                elif s[-1] == ".":
                    last_sentences.append(s)
                else:
                    last_sentences.append(s + "?")
            except Exception as e:
                print(s)
    sentences = last_sentences
    x = 0
    for sent in sentences:
        if clean_string:
            orig_rev = sent.strip()
            if orig_rev == '':
                continue
            words = set(orig_rev.split())
            splitted = orig_rev.split()
            x += len(splitted)
            if len(splitted) > 200:
                # chunk huge sentences to small ones.
                orig_rev = []
                splits = int(np.floor(len(splitted) / 200))
                for index in range(splits):
                    orig_rev.append(' '.join(splitted[index * 200:(index + 1) * 200]))
                if len(splitted) > splits * 200:
                    orig_rev.append(' '.join(splitted[splits * 200:]))
                status.extend(orig_rev)
            else:
                status.append(orig_rev)
        else:
            orig_rev = sent.strip().lower()
            words = set(orig_rev.split())
            status.append(orig_rev)

        for word in words:
            vocab[word] += 1

    datum = {"text": status,
             "num_words": np.max([len(sent.split()) for sent in status])}
    revs.append(datum)
    m.append(x)
    print(len(revs))
    return revs, vocab

def text_preprocessing_2(rev_list, clean_string=True):
    revs = []
    vocab = defaultdict(float)
    for rev in rev_list:
        for line in rev["text"]:
            status = []
            sentences = line.strip()
            if clean_string:
                orig_rev = clean_str(sentences.strip())
                words = set(orig_rev.split())
                splitted = orig_rev.split()

                if len(splitted) > 250:
                    # chunk huge sentences to small ones.
                    orig_rev = []
                    splits = int(np.floor(len(splitted) / 250))
                    for index in range(splits):
                        orig_rev.append(' '.join(splitted[index * 250:(index + 1) * 250]))
                    if len(splitted) > splits * 250:
                        orig_rev.append(' '.join(splitted[splits * 250:]))
                    status.extend(orig_rev)
                else:
                    status.append(orig_rev)
            else:
                orig_rev = sentences.strip().lower()
                words = set(orig_rev.split())
                status.append(orig_rev)

            for word in words:
                vocab[word] += 1

            datum = {"text": status,
                     "num_words": np.max([len(sent.split()) for sent in status])}
            revs.append(datum)
    return revs, vocab
        
def load_bert_vec(revs):
    start_time = time.time()
    now_time = time.time()
    bc = BertClient()

    for rev_idx, rev in enumerate(revs):
        rev_splitted = [orig_rev.split() for orig_rev in rev["text"]]
        print(str((100 * rev_idx + 0.0) / len(revs)) + "% done")
        print(str(time.time() - now_time) + "sec passed")
        print(str((time.time() - start_time) * ((len(revs) - rev_idx) / (rev_idx + 1))) + "sec need to to complete")
        print(str((time.time() - start_time) * (len(revs) / (rev_idx + 1))) + "sec need in total")
        result = bc.encode(rev_splitted, is_tokenized=True)
        rev["embedding"] = result

def prediction(revs, vocab, l=1):
    for rev in revs:
        if l != 0:
            rev["embedding"] = np.mean(rev["embedding"][:, (l - 1) * 768:l * 768], axis=0)
        else:
            rev["embedding"] = np.mean(rev["embedding"][:, -768 * 4:], axis=0)
    X_test = [rev["embedding"]]
    filename = 'finalized_model_run_1_Y0.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(X_test)
    print(result)
    

if __name__ == "__main__":
#     text = """Romans countrymen and friends, listen to what I have to say and be silent so that you can hear. Trust me for my honour and show respect so that you will follow what I say. Judge me according to your wisdom and use your understanding so that you will be able to judge better. If there is anyone in this assembly, any dear friend of Caesar’s, to him I say that Brutus’ love for Caesar was no less than his. If then that dear friend demands to know why Brutus rose against Caesar, this is my answer – not that I loved Caesar less but that I loved Rome more. Would you rather Caesar were living, and all die slaves, than that Caesar were dead, to all live as free men? As Caesar loved me, I weep for him; as he was fortunate, I rejoice at it; as he was brave, I honour him; but as he was ambitious, I killed him. There are tears for his love; joy for his fortune; honour for his valour; and death for his ambition. Is there anyone here so lacking in pride that we wants to be a slave? If there is, speak, because it’s he I have offended. Who is here so low that he doesn’t want to be a Roman? If any, speak, for it’s him I have offended. Who is here so vile that he does not love his country? If any, speak, for him I have offended. I have done no more to Caesar than you would do to Brutus. The things that Caesar died for are recorded in the Capitol. His glory, for which he was renowned, is not understated; not his offences exaggerated, for which he suffered death
# Here comes his body, mourned by Mark Antony, who, although he had no hand in Caesar’s death, will receive the benefit of his dying – a place in the commonwealth, as which of you won’t? With this I leave you: that as I slew my best friend for the good of Rome, I have the same dagger for myself when it shall please my country to need my death."""
    
    text = """Friends, Romans, countrymen, give me your attention. I have come here to bury Caesar, not to praise him. The evil that men do is remembered after their deaths, but the good is often buried with them. It might as well be the same with Caesar. The noble Brutus told you that Caesar was ambitious. If that’s true, it’s a serious fault, and Caesar has paid seriously for it. With the permission of Brutus and the others—for Brutus is an honorable man; they are all honorable men—I have come here to speak at Caesar’s funeral. He was my friend, he was faithful and just to me. But Brutus says he was ambitious, and Brutus is an honorable man. He brought many captives home to Rome whose ransoms brought wealth to the city. Is this the work of an ambitious man? When the poor cried, Caesar cried too. Ambition shouldn’t be so soft. Yet Brutus says he was ambitious, and Brutus is an honorable man. You all saw that on the Lupercal feast day I offered him a king’s crown three times, and he refused it three times. Was this ambition? Yet Brutus says he was ambitious. And, no question, Brutus is an honorable man. I am not here to disprove what Brutus has said, but to say what I know. You all loved him once, and not without reason. Then what reason holds you back from mourning him now? Men have become brutish beasts and lost their reason! Bear with me. My heart is in the coffin there with Caesar, and I must pause until it returns to me. """
    revs_1, vocab_1 = text_preprocessing_1(text)
    revs_2, vocab_2 = text_preprocessing_2(revs_1)
    load_bert_vec(revs_2)
    prediction(revs_2, vocab_2)
    