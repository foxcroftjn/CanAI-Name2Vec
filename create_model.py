from common import get_surnames
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def train_model(epochs, vector_size, window):
    documents = [TaggedDocument(list(doc), [i]) for i, doc in enumerate(get_surnames())]
    model = Doc2Vec(documents, epochs=epochs, vector_size=vector_size, window=window, workers=1)
    return model
