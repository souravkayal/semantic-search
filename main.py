from fastapi import FastAPI, HTTPException, status
from product import Product, ProductSearch
from product_vector import ProductTransformer
from product_repository import ProductRepository
from transformers import pipeline


app = FastAPI()

# check how to show product name and description
# check hos to reduce embed generating time
# test with more data
# implement RAG based use-case


@app.post("/product/add")
def add_product(product: Product):

    try:
        product_vector_result = ProductTransformer().product_encode(product)
        metadata = {"name": product.name, "description": product.description}
        return ProductRepository().save_product(product_vector_result["product_id"],
                                                product_vector_result["embedding"],
                                                metadata)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/product/search")
def add_product(product: ProductSearch):
    generator = pipeline('text-generation', model='gpt2')

    try:
        product_vector_result = ProductTransformer().encode_description(product.description)
        search_result = ProductRepository().read_product(product_vector_result)

        for result in search_result["results"]:
            prompt = f"Based on the following product descriptions, provide a summary:\n\n{product.description}\n\nSummary:"
            response = generator(prompt, max_length=150,
                                 num_return_sequences=1,
                                 truncation=True)

            text_desc = response[0]['generated_text'].strip()
            result["summary"] = text_desc

        return search_result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
