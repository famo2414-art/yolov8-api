bind = ":8000"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
threads = 2
timeout = 120
accesslog = "-"
errorlog = "-"
