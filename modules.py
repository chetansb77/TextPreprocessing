import unidecode
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re
from spellchecker import SpellChecker
from deepsegment import DeepSegment
from pycontractions import Contractions
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from nltk import Tree
import preprocessor as p

# Replace multiple ... to single .
# Takes text in the form of string as input and returns output as string
def preformats(strText):
    x = re.sub(r'\. +', ".", strText)
    x = re.sub(r'\.\.+', ". ", x)
    return x

# Replace Accented Characters
# Takes text in the form of string as input and returns output as string with accented characters replaced
def replace_accented_characters(strText):
    unaccented_string = unidecode.unidecode(strText)
    return unaccented_string


# Sentence Tokenizer
# Takes the text in the form of string as input and returns output as a list of sentences
def sentence_tokenizer(strText):
    sentTokenizeList = sent_tokenize(strText)
    return sentTokenizeList


# Expands contractions
# Takes the text in the form of string as input and returns output as string with expanded contractions
def expand_contractions(strText):
    cont = Contractions(api_key="glove-twitter-100")
    expandedText = list(cont.expand_texts([strText], precise=True))
    return expandedText[0]


# Irrelevent Text Removal
# This function removes texts in sentences such as #tagexts, URLs etc
# Takes the text in the form of string as input and returns output as string with irrelevent text removed
def remove_irrelevent_text(strText):
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG)

    x = strText
    x = p.clean(x)
    x = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", x).split())
    x = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''," ", x)
    # if x[-1] == '.':
    #     x = x[:-1]

    return x


# Removes Additional White Spaces
# Takes the text in the form of string as input and returns output as string with additional white spaces removed
def remove_white_spaces(strText):
    x = " ".join(strText.split())
    return x


# Spellcheck
# Autocorrects misspelled words
# Takes the text in the for of string as input and returns output as string with corrected misspelled words
def spellcheck(strText):
    spell = SpellChecker()
    nltk.download('punkt')

    textWordlist = word_tokenize(strText)

    # find those words that may be misspelled
    misspelled = spell.unknown(textWordlist)

    for word in misspelled:
        # Get the one `most likely` answer
        index = textWordlist.index(word)
        textWordlist[index] = spell.correction(word)

    x = " ".join(textWordlist)
    return x


# Simplify Unpunctuated sentences
# Divides into simple sentences for unpunctuated sentences
# Takes the text in the form of string as input and returns a list of divided sentences
segmenter = DeepSegment('en')

def simplify_unpunctuated(strText):
    dividedSent = segmenter.segment(strText)
    return dividedSent


def capitalizeFirstLetter(strTrext):
    word = strTrext.split()
    if len(word) > 0:
        word[0] = word[0].title();
        x = " ".join(word)
        return x
    else:
        return strTrext

# Converts Trees to Dictionary
# Takes Tress in the form of string and returns dictionary
def tree_to_dict(tree):
    dictTree = Tree.fromstring(tree)
    return dictTree


# Setup AllenNLP Constituency Parser
archive = load_archive(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz"
)
predictor = Predictor.from_archive(archive, 'constituency-parser')


def get_child_node_tags(treeDict):
    subtreesLen = treeDict.__len__()
    subtreeTagList = []
    for i in range(subtreesLen):
        subtreeTagList.append(treeDict[i]._label)

    return subtreeTagList


def getSubclauses(parsedDictTree):
    clauseTags = ['S', 'SBAR']
    subtreesLen = parsedDictTree.__len__()
    clauseList = []
    for i in range(subtreesLen):
        if parsedDictTree[i]._label in clauseTags:
            subclause = Tree.flatten(parsedDictTree[i])
            subclause = ' '.join(subclause)
            clauseList.append(subclause)

    return clauseList


def divideSubsentence(parsedDictTree):
    clauseTags = ['S', 'SBAR']
    conjunctionTags = ['CC', ',']

    subtreesLen = parsedDictTree.__len__()
    clauseList = []
    sentCount = 0
    clauseList = []
    str = ""
    for i in range(subtreesLen):
        if parsedDictTree[i]._label not in clauseTags and parsedDictTree[i]._label not in conjunctionTags:
            subclause = Tree.flatten(parsedDictTree[i])
            subclause = ' '.join(subclause)
            str = str + " " + subclause

        if parsedDictTree[i]._label in conjunctionTags:
            clauseList.append(str.strip())
            str = ""

        if parsedDictTree[i]._label in clauseTags:
            clauseList.append(str.strip())
            str = ""
            subclause = Tree.flatten(parsedDictTree[i])
            subclause = ' '.join(subclause)
            clauseList.append(subclause)

    if len(str) > 0:
        clauseList.append(str.strip())
        str = ""

    while ("" in clauseList):
        clauseList.remove("")

    return clauseList

# Divide Sentences At Conjunctions
# Takes the text in the form
def complex_to_simple_sentence(strText):
    parsedTreeString = predictor.predict_json({"sentence": strText})
    parsedDictTree = tree_to_dict(parsedTreeString['trees'])
    childTagList = get_child_node_tags(parsedDictTree)

    conjunctionTags = ['CC', ',']
    clauseTags = ['S', 'SBAR']
    simplefiedTagList = []
    isDividable = []
    subclauses = []

    for tag in childTagList:
        if tag not in conjunctionTags:
            simplefiedTagList.append(tag)
        else:
            if len(simplefiedTagList) == 0 or simplefiedTagList[-1] != 'CC':
                simplefiedTagList.append('CC')
            else:
                continue

    if len(simplefiedTagList) % 2 != 0:
        for index, tag in enumerate(simplefiedTagList):
            if index % 2 == 0:
                if tag in clauseTags:
                    isDividable.append(True)
                else:
                    isDividable.append(False)
            else:
                if tag in conjunctionTags:
                    isDividable.append(True)
                else:
                    isDividable.append(False)


    if False in isDividable or len(isDividable) == 0:
        if any(clause in clauseTags for clause in childTagList) and any(conjunction in conjunctionTags for conjunction in childTagList):
            subclauses = divideSubsentence(parsedDictTree)
        else:
            subclauses.append(strText)
    else:
        subclauses = getSubclauses(parsedDictTree)

    return subclauses
