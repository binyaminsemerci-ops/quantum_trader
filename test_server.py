from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="debug")