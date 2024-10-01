from langchain_community.embeddings import JinaEmbeddings


def get_embedding_function():
    embeddings = JinaEmbeddings(
        model_name="jina-embeddings-v2-base-zh",
    )

    return embeddings
