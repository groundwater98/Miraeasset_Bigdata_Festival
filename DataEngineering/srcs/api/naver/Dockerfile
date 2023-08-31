FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY api/naver/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./api ./api
COPY ./tools ./tools
COPY setup.py .
COPY setup.sh .
RUN chmod +x ./setup.sh

ENTRYPOINT ["./setup.sh"]

# Command to run your application
# CMD [ "ls", "/app" ]
CMD [ "python", "api/naver/main.py" ]