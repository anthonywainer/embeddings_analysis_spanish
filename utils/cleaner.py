import json
import re
import requests
import nltk

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def get_words(text_content):
    """ Function to get words from text content

        @param text_content    The text content
    """
    text_in_lower = str(text_content).replace('\n', ' ').replace("\t", " ").lower()

    emoji_text = emoji.replace_emoji(text_in_lower)

    only_text = re.sub(
        r'(#[A-Za-z0-9_]+)|(.#[A-Za-z0-9_]+)|(@[A-Za-z0-9_]+)|(.@[A-Za-z0-9_]+)|(http\S+)|([^a-záéíóúñ ]+)', ' ',
        emoji_text)
    dropped_words = set(only_text.split())
    return list(dropped_words)


def processing_words(text_content, own_words=None, lang="spanish"):
    """ Function to processing words from text content

        :param lang: language
        :param text_content    The text content
        :param own_words       Words customized
    """

    if own_words is None:
        own_words = []
    words = get_words(text_content)

    spanish_stops = set(stopwords.words(lang) + own_words)

    return " ".join([word for word in words if word not in spanish_stops and len(word) > 3])


class EmojiDecode:

    def __init__(self, emoji_path):
        emoji_file = requests.get(emoji_path)
        self.emojis = json.loads(emoji_file.text)

    @staticmethod
    def multiple_replace(di, text):
        regex = re.compile("(%s)" % "|".join(map(re.escape, di.keys())))

        return regex.sub(lambda mo: di[mo.string[mo.start():mo.end()]], text)

    def filter_emojis(self, by_replace):
        return {i: self.emojis[i] for i in by_replace if i in self.emojis.keys()}

    def replace_emoji(self, text):
        by_replace = list(set(re.findall(r'[^\w\s,]', text)))
        if not by_replace:
            return text

        emojis = self.filter_emojis(by_replace)
        if not emojis:
            return text

        return self.multiple_replace(emojis, text)


__URL__ = 'https://github.com/anthonywainer/embeddings_analysis_spanish/blob/master/dataset/emoticones.json?raw=true'
emoji = EmojiDecode(__URL__)
