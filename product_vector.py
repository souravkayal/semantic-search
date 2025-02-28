from sentence_transformers import SentenceTransformer


class ProductTransformer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def product_encode(self, product):
        product_text = f"{product.name} {product.description}"
        embedding = self.model.encode(product_text)
        return {"product_id": str(product.id), "embedding": embedding.tolist()}

    def encode_description(self, description):
        return self.model.encode(description)
