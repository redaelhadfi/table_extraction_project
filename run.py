# run.py
import uvicorn

if __name__ == "__main__":
    # It's good practice to specify the app string as "module:instance"
    # reload=True is great for development
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
