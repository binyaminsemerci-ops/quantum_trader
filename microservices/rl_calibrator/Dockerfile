FROM python:3.10-slim
WORKDIR /app
COPY microservices/rl_calibrator/calibrate_tp_sl.py /app/calibrate_tp_sl.py
RUN pip install --no-cache-dir redis torch numpy scipy
CMD ["python3", "-u", "calibrate_tp_sl.py"]
