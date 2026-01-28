# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "agent.py"]

# docker-compose.yml
version: '3.8'
services:
  {design.agent_name.lower().replace(' ', '-')}:
    build: .
    environment:
      - OPENAI_API_KEY=${{OPENAI_API_KEY}}
      - ANTHROPIC_API_KEY=${{ANTHROPIC_API_KEY}}
    ports:
      - "8000:8000"