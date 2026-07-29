FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY model/onnx/model.onnx ${LAMBDA_TASK_ROOT}/model/onnx/
COPY model/tokenizer.json ${LAMBDA_TASK_ROOT}/model/
COPY app.py     ${LAMBDA_TASK_ROOT}/

# LAMBDA_TASK_ROOT is /var/task; the handler is module.function
CMD [ "app.handler" ]