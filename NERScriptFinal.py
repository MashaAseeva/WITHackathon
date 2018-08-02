
# coding: utf-8

# In[8]:


def add_event_ent(matcher, doc, i, matches):
    EVENT = nlp.vocab.strings['EVENT']
    # Get the current match and create tuple of entity label, start and end.
    # Append entity to the doc's entity. (Don't overwrite doc.ents!)
    match_id, start, end = matches[i]
    entity = (EVENT, start, end)
    doc.ents += (entity,)

#trying nltk for named entity recognition
import spacy, numpy
from spacy.symbols import ORTH, LEMMA, POS, TAG, ENT_TYPE, ENT_IOB
from spacy.matcher import Matcher
import pandas as pd
import operator

nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

trainingVariants= pd.read_csv('training_variants', sep=",", engine="python")
trainingY = trainingVariants.drop(['Class'], axis=1)

# construct the pattern
pattern = []
for i in set(trainingY['Gene']):
    newDict = [{}]
    newDict[0]['ORTH'] = i
    pattern.append(newDict)
    
matcher.add('Gene', add_event_ent, *pattern)

pattern2 = []
for i in set(trainingY['Variation']):
    newDict = [{}]
    newDict[0]['ORTH'] = i
    pattern2.append(newDict)
    
matcher.add('Variant', add_event_ent, *pattern2)

doc = nlp(u'Testing this for gene BRCA1')

#now we are running it on the training text
trainingText = pd.read_csv('training_text', sep="\|\|", engine='python', skiprows=1, names=["ID","Text"])
trainingX = trainingText
trainingX = trainingX[trainingX['Text'] != 'null']
trainingY = trainingY[trainingY.ID.isin(trainingX['ID'])]

print("about to start loop")

accuracyGene = 0
accuracyVar = 0
#decision for the content: take majority entity value
for index, row in trainingX.iterrows():
    doc = nlp(row['Text'])
    matches = matcher(doc)
    geneDict = {}
    variantDict = {}
    geneprediction = 0
    varprediction = 0
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        if string_id == 'Gene':
            span = doc[start:end]
            if span.text in geneDict:
                geneDict[span.text] +=1
            else:
                geneDict[span.text] = 1
        elif string_id == 'Variant':
            span = doc[start:end]
            if span.text in variantDict:
                variantDict[span.text] +=1
            else:
                variantDict[span.text] = 1
                
#     print(geneDict, variantDict)
    if len(geneDict) > 0:
        geneprediction = max(geneDict.items(), key=operator.itemgetter(1))[0]
#         print("Gene prediction:", geneprediction)
    if len(variantDict) > 0:
        varprediction = max(variantDict.items(), key=operator.itemgetter(1))[0]
#         print("Variant prediction:", varprediction)
    
#     print(index)

    
    if trainingY.iloc[index]['Gene'] == geneprediction:
        accuracyGene += 1
        
    if trainingY.iloc[index]['Variation'] == varprediction:
        accuracyVar += 1
        
    if index > 100:
        break

print("Gene accuracy:", accuracyGene/100)
print("Variation accuracy:", accuracyVar/100)

# print("Gene accuracy:", accuracyGene/len(trainingY))
# print("Variation accuracy:", accuracyVar/len(trainingY))


# In[ ]:


testText = pd.read_csv('test_text', sep="\|\|", engine='python', header=None, names=["ID","Text"])
testVariants = pd.read_csv('test_variants', sep=",", engine = "python", header=None, names=["ID","Gene","Variation","Class"])


testingX = testText
testingY = testVariants.drop(['Class'], axis=1)

testingX = testingX[testingX['Text'] != 'null']
testingY = testingY[testingY.ID.isin(testingX['ID'])]


for index, row in testingX.iterrows():
    doc = nlp(row['Text'])
    matches = matcher(doc)
    geneDict = {}
    geneprediction = 0
    varprediction = 0
    variantDict = {}
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]
        if string_id == 'Gene':
            span = doc[start:end]
            if span.text in geneDict:
                geneDict[span.text] +=1
            else:
                geneDict[span.text] = 1
        elif string_id == 'Variant':
            span = doc[start:end]
            if span.text in variantDict:
                variantDict[span.text] +=1
            else:
                variantDict[span.text] = 1
                
#     print(geneDict, variantDict)
    if len(geneDict) > 0:
        geneprediction = max(geneDict.items(), key=operator.itemgetter(1))[0]
#         print("Gene prediction:", geneprediction)
    if len(variantDict) > 0:
        varprediction = max(variantDict.items(), key=operator.itemgetter(1))[0]
#         print("Variant prediction:", varprediction)
    
#     print(index)

    
    if testingY.iloc[index]['Gene'] == geneprediction:
        accuracyGene += 1
        
    if testingY.iloc[index]['Variation'] == varprediction:
        accuracyVar += 1
        
print("Gene accuracy:", accuracyGene/len(testingY))
print("Variation accuracy:", accuracyVar/len(testingY))

