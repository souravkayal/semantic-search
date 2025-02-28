from fastapi import FastAPI, HTTPException, status
from product import Product, ProductSearch
from product_vector import ProductTransformer
from product_repository import ProductRepository

app = FastAPI()

# swagger integration
# check how to show product name and description
# check hos to reduce embed generating time
# test with more data
# implement RAG based use-case


@app.post("/product/add")
def add_product(product: Product):

    try:
        product_vector_result = ProductTransformer().product_encode(product)
        return ProductRepository().save_product(product_vector_result["product_id"],
                                                product_vector_result["embedding"])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/product/search")
def add_product(product: ProductSearch):

    try:
        product_vector_result = ProductTransformer().encode_description(product.description)
        return ProductRepository().read_product(product_vector_result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
