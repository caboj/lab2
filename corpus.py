import numpy as np
import nltk

class Collection:
    def __init__(self, files):
        self.tasks = []
        self.vocab = []
        self.vectors = {}
        for fileName in files:
            self.read_task(fileName)

    def read_task(self, fileName):
        self.tasks.append(Task(fileName))

    def translate(self):
        vocab = self.getVocabulary()
        for t in self.tasks:
            t.translate(vocab)

    def getVocabulary(self):
        if not len(self.vocab):
            words = []
            for t in self.tasks:
                words += t.getWords()
            self.vocab = np.concatenate((['<BOS>','<EOS>'], np.unique(words)))
        return self.vocab

    def getVectors(self, reverse=True, translated=False):
        key = str((reverse,translated))
        if key not in self.vectors:
            vecs = {'input':[],'output':[]}
            for t in self.tasks:
                if reverse:
                    tvec = t.getReverseVectors(translated)
                else:
                    tvec = t.getQuestionVectors(translated)
                #vecs['input'] = np.concatenate((vecs['input'], tvec['input']))
                #vecs['output'] = np.concatenate((vecs['output'], tvec['output']))
                vecs['input'] += tvec['input']
                vecs['output'] += tvec['output']
            self.vectors[key] = vecs
        return self.vectors[key]

    def printInfo(self):
        print('\nCollection built with file(s):')
        for i, t in enumerate(self.tasks):
            print('('+str(i+1)+')\t'+t.fileName)
            t.printInfo()
        print('\n')

    def __str__(self):
        string = ''
        for t in self.tasks:
            string += '\n'+str(t)
        return string

class Task:
    def __init__(self, fileName):
        self.fileName = fileName
        
        self.stories = []
        self.read_stories()
        
        self.translated = False

    def read_stories(self):
        text = []
        nst = 0 # set to positive number to limit number of stories
        with open(self.fileName, 'rU') as f:
            for i, line in enumerate(f.readlines()):
                s = np.array(nltk.word_tokenize(line.strip()))
                if s[0]=='1':
                    if text:
                        self.stories.append(Story(text))
                        text = []
                        if nst and (len(self.stories)==nst):
                            print('BREAK')
                            break
                text.append(s)
        if text:
            self.stories.append(Story(text))

    def getWords(self):
        words = []
        for st in self.stories:
            words += st.getWords()
        return words

    def translate(self, vocab):
        print('Translating...')
        for story in self.stories:
            story.translate(vocab)
        self.translated = True

    def getReverseVectors(self, translated=False):
        vecs = {'input':[],'output':[]}
        textType = 'translation' if translated else 'text'
        (bos, eos) = ([0], [1]) if translated else (['<BOS>'], ['<EOS>'])
        for st in self.stories:
            utterances = st.utterances
            for i in range(len(utterances)):
                if utterances[i].uType == 'answer':
                    continue
                elif utterances[i].uType == 'question':
                    q = getattr(utterances[i], textType)
                    a = getattr(utterances[i+1], textType)
                    v = np.concatenate((q, a))
                else:
                    v = getattr(utterances[i], textType)
                vecs['input'].append(np.concatenate((bos, v, eos)))
                vecs['output'].append(np.concatenate((bos, v[::-1], eos)))
        return vecs

    def getQuestionVectors(self, translated=False):
        vecs = {'input':[],'output':[]}
        textType = 'translation' if translated else 'text'
        (bos, eos) = ([0], [1]) if translated else (['<BOS>'], ['<EOS>'])
        for st in self.stories:
            context = []
            for ut in st.utterances:
                v = getattr(ut, textType)
                if ut.uType == 'answer':
                    vecs['output'].append(np.concatenate((bos,v,eos)))
                elif ut.uType == 'question':
                    # The order of concatenation matters, the question should come first
                    vecs['input'].append(np.concatenate((bos,v,context,eos)))
                else:
                    context = np.concatenate((context, v))
        return vecs

    def printInfo(self):
        print('# of stories:\t  '+str(len(self.stories)))
        utn = sum([len(st.utterances) for st in self.stories])
        print('# of utterances:  '+str(utn))

    def __str__(self):
        string = 'TASK: '+self.fileName+'\n'
        for i, st in enumerate(self.stories):
            string += '\nSTORY '+str(i+1)+'\n'+str(st)
        return string

class Story:
    def __init__(self, text):
        self.utterances = []
        self.process_text(text)

    def process_text(self, text):
        self.utterances = []
        for s in text:
            if '.' in s:
                self.utterances.append(Utterance(s[1:], 'statement'))
            elif '?' in s:
                qid = np.where(s=='?')[0][0]
                self.utterances.append(Utterance(s[1:(qid+1)], 'question'))
                self.utterances.append(Utterance(s[(qid+1):(qid+2)], 'answer'))
   
    def getWords(self):
        words = []
        for ut in self.utterances:
            words += list(ut.text)
        return words 

    def translate(self, vocab):
        for utter in self.utterances:
            utter.translate(vocab)

    def __str__(self):
        string = ''
        for u in self.utterances:
            string += '\n'+str(u)
        return string

class Utterance():
    def __init__(self, text, uType):
        self.text = text
        self.uType = uType
        self.translation = []

    def translate(self, vocab):
        vIds = np.where(vocab==np.array_split(self.text, len(self.text)))[1]
        self.translation = vIds

    def __str__(self):
        string = 'type: '+ str(self.uType)
        string += '\ntext: '+ str(self.text)
        string += '\ntranslation: '+ (str(self.translation) if len(self.translation) else 'nav')
        string += '\n'
        return string