from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd
import modules as mods
import multiprocessing
import time
from tqdm import tqdm

reviewList = []
resolvedlist = []
rawReviewList = []


start = time.time()
data = pd.read_excel('processed_review.xlsx')
# reviewList = data['reviews']
reviewList = ["I love these Kind bars, they are my absolute favorite, and when I saw what a good price a big box was compared to the grocery store, I had to buy them. However, these taste terrible. They smell bad, almost chemical, and leave a disgusting after taste that will not go away. I thought maybe it was just me so I tried another. Same thing. I left them for a few days and tried once more and they were so gross. Immediately upon opening the bar wrapper you smell this scent almost akin to gasoline. I am so disappointed, I had to throw the whole box out. Save your tongue and buy from your grocery store."]

def textPreprocessing(strInput):
    # strInput = mods.replace_accented_characters(strInput)
    strInput = mods.preformats(strInput)
    tokenizedReview = mods.sentence_tokenizer(strInput)

    resolvedReview = []

    for index, sentence in enumerate(tokenizedReview):
        formattedSentence = sentence
        formattedSentence = mods.expand_contractions(formattedSentence)
        formattedSentence = mods.remove_irrelevent_text(formattedSentence)
        formattedSentence = mods.remove_white_spaces(formattedSentence)
        # formattedSentence = mods.spellcheck(formattedSentence)
        # deepsegmentedSentence = mods.simplify_unpunctuated(formattedSentence)
        deepsegmentedSentence = [formattedSentence]

        for smlSent in deepsegmentedSentence:
            simpleSentences = mods.complex_to_simple_sentence(smlSent)
            for sent in simpleSentences:
                sent = mods.capitalizeFirstLetter(sent)
                resolvedReview.append(sent)

    return ' . '.join(resolvedReview)


for i, review in tqdm(enumerate(reviewList)):
    try:
        processedReview = textPreprocessing(review)
        processedReview = mods.preformats(processedReview)
        rawReviewList.append(review)
        resolvedlist.append(processedReview)
        # print(review)
        # print(processedReview)
    except:
        print(review)
        rawReviewList.append(review)
        resolvedlist.append(review)

resultData = {
    'Review': rawReviewList,
    'ResolvedList': resolvedlist
}

done = time.time()
elapsed = done - start
print(elapsed)
df = pd.DataFrame(resultData)

writer = pd.ExcelWriter('TextPreprocessed.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
writer.save()