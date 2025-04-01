FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

# Run all scripts in src
CMD ["sh", "-c", "python /app/src/SubgroupA/1.0-dtth-guest-satisfaction-factors.py && \
                 python /app/src/SubgroupA/2.0-alsy-guest-segmentation-model.py && \
                 python /app/src/SubgroupA/3.0-yz-guest-journey-patterns.py && \
                 python /app/src/SubgroupA/4.0-ic-marketing-strategies-guest-behaviour.py && \
                 python /app/src/SubgroupA/5.0-bsx-external-factors-guest-segmentation.py && \
                 python /app/src/SubgroupB/6.0-cyjw-demand-prediction-attractions-services.py && \
                 python /app/src/SubgroupB/7.0-bwt-optimization-layout-and-schedule.py && \
                 python /app/src/SubgroupB/8.0-fwx-resource-allocation-demand-variability.py && \
                 python /app/src/SubgroupB/9.0-lwhj-predicting-guest-complaints-service-recovery.py && \
                 python /app/src/SubgroupB/10.0-tjlj-IOT_data_integration_experience_optimization.py"]

