import re


def remove_stopwords(item, usage):
    stop_words = f'当作|当|用|{item}'
    tmp = re.sub(stop_words, '', usage)
    if tmp:
        return re.sub(item, '', usage)
    else:
        return usage

