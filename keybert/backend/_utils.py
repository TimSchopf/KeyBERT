import nltk
import spacy
from nltk.corpus import stopwords

from ._base import BaseEmbedder
from ._sentencetransformers import SentenceTransformerBackend


def select_backend(embedding_model) -> BaseEmbedder:
    """ Select an embedding model based on language or a specific sentence transformer models.
    When selecting a language, we choose `all-MiniLM-L6-v2` for English and
    `paraphrase-multilingual-MiniLM-L12-v2` for all other languages as it support 100+ languages.

    Returns:
        model: Either a Sentence-Transformer or Flair model
    """
    # keybert language backend
    if isinstance(embedding_model, BaseEmbedder):
        return embedding_model

    # Flair word embeddings
    if "flair" in str(type(embedding_model)):
        from keybert.backend._flair import FlairBackend
        return FlairBackend(embedding_model)

    # Spacy embeddings
    if "spacy" in str(type(embedding_model)):
        from keybert.backend._spacy import SpacyBackend
        return SpacyBackend(embedding_model)

    # Gensim embeddings
    if "gensim" in str(type(embedding_model)):
        from keybert.backend._gensim import GensimBackend
        return GensimBackend(embedding_model)

    # USE embeddings
    if "tensorflow" and "saved_model" in str(type(embedding_model)):
        from keybert.backend._use import USEBackend
        return USEBackend(embedding_model)

    # Sentence Transformer embeddings
    if "sentence_transformers" in str(type(embedding_model)):
        return SentenceTransformerBackend(embedding_model)

    # Create a Sentence Transformer model based on a string
    if isinstance(embedding_model, str):
        return SentenceTransformerBackend(embedding_model)

    return SentenceTransformerBackend("paraphrase-multilingual-MiniLM-L12-v2")


def remove_suffixes(text: str, suffixes:list) -> str:
    """Removes pre-defined suffixes from a given text string.

    Arguments:
        text: Text string where suffixes should be removed
        suffixes: List of strings that should be removed from the end of the text

    Returns:
        text: Text string with removed suffixes
    """
    for suffix in suffixes:
        if text.lower().endswith(suffix.lower()):
             return text[:-len(suffix)].strip()
    return text


def remove_prefixes(text: str, prefixes: list) -> str:
    """Removes pre-defined prefixes from a given text string.

    Arguments:
        text: Text string where prefixes should be removed
        suffixes: List of strings that should be removed from the beginning of the text

    Returns:
        text: Text string with removed prefixes
    """
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):].strip()
    return text

def get_candidate_keyphrases(document: str, stop_words: str) -> list:
    """Select candidate keyphrases with POS-tagging from a given text document.
    Only select keyphrases that have 0 or more adjectives, followed by 0 or more nouns as candidate keyphrases.
    Optionally, remove unwanted stopwords from keyphrases if 'stop_words' is not None.

    Arguments:
        document: The document for which to extract the candidate keyphrases
        stop_words: Language of stopwords to remove from the document e.g. 'english'

    Returns:
        candidate_keyphrases: List of unique noun-keyphrases of varying length, extracted from the given document
    """

    stop_words_list = []
    if stop_words:
        stop_words_list = set(stopwords.words(stop_words))

    # add SpaCy POS tags for document
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    tagged_doc = nlp(document)
    tagged_pos_doc = []
    for sentence in tagged_doc.sents:
        pos_tagged_sentence = []
        for word in sentence:
            pos_tagged_sentence.append((word.text, word.tag_))
        tagged_pos_doc.append(pos_tagged_sentence)

    # extract keyphrases that match the NLTK RegexpParser filter
    cp = nltk.RegexpParser('CHUNK: {(<J.*>*<N.*>*)}')
    candidate_keyphrases = []
    prefix_list = [stop_word + ' ' for stop_word in stop_words_list]
    suffix_list = [' ' + stop_word for stop_word in stop_words_list]
    for sent in tagged_pos_doc:
        tree = cp.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'CHUNK':
                # join candidate keyphrase from single words and convert it to lower case
                keyphrase = ' '.join([i[0] for i in subtree.leaves()]).lower()

                # remove stopword suffixes
                keyphrase = remove_suffixes(keyphrase, suffix_list)

                # remove stopword prefixes
                keyphrase = remove_prefixes(keyphrase, prefix_list)

                # remove whitespace from the beginning and end of keyphrases
                keyphrase = keyphrase.strip()

                # do not include single keywords that are actually stopwords
                if keyphrase not in stop_words_list:
                    candidate_keyphrases.append(keyphrase)

    # remove potential empty keyphrases
    candidate_keyphrases = [keyphrase for keyphrase in candidate_keyphrases if keyphrase != '']

    return list(set(candidate_keyphrases))