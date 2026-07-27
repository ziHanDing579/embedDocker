FROM public.ecr.aws/lambda/python:3.12

# Deps installed into the Lambda task root
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Your code + baked-in model weights
COPY app.py ${LAMBDA_TASK_ROOT}
COPY model/ ${LAMBDA_TASK_ROOT}/model/

# handler = "<filename>.<function name>"
CMD ["app.handler"]