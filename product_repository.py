from pinecone import Pinecone
import numpy as np


class ProductRepository:

    def __init__(self):
        self.pinecone_index_name = "products"
        self.api_key = "pcsk_4LMuph_SyX5Fvp8rdpoQL5exrg2TZDrNCg2oMDdncT2sgcqE1nBT9PFPjBtmrKsBEJfMmp"
        self.pinecone_namespace = "ns1"
        self.index = Pinecone(self.api_key).Index(self.pinecone_index_name)

    def save_product(self, id, embeddings):

        try:
            self.index.upsert(
                vectors=[
                    {
                        "id": id,
                        "values": embeddings
                    }
                ],
                namespace=self.pinecone_namespace
            )
            return {"status": "success"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def read_product(self, descriptionEmbedding):
        try:

            if isinstance(descriptionEmbedding, np.ndarray):
                description_embedding = descriptionEmbedding.tolist()

            response = self.index.query(vector=description_embedding,
                                        top_k=5,
                                        namespace=self.pinecone_namespace)

            results = [{"id": match["id"], "score": match["score"]}
                       for match in response["matches"]]

            return {"status": "success", "results": results}

        except Exception as e:
            return {"status": "error", "message": str(e)}
