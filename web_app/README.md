# Bonus Web Application Frontend

This folder contains a free local Flask web app for the NewsBot 2.0 final project.

It does not require paid APIs, API keys, or paid hosting. The app trains lightweight local models from the BBC News Classification dataset and lets a user analyze articles in a browser.

## Features

- Article text input form
- Category prediction with confidence
- Sentiment analysis
- Extractive summary
- Named entity extraction
- Topic keywords
- Similar article search
- Insight list
- JSON API endpoint
- Downloadable JSON results

## How to run locally

1. Make sure `learn-ai-bbc.zip` is available in one of these places:
   - `web_app/data/learn-ai-bbc.zip`
   - project root: `ITAI2373-NewsBot-Final/learn-ai-bbc.zip`
   - your Downloads folder

2. Install requirements from the project root:

```bash
pip install -r requirements.txt
```

3. Start the Flask app:

```bash
cd web_app
python app.py
```

4. Open this address in your browser:

```text
http://127.0.0.1:5000
```

The first run may take a moment because the app loads the dataset and trains local NLP models.

## API endpoint

The app also provides a JSON endpoint:

```text
POST /api/analyze
```

Example request body:

```json
{
  "text": "Microsoft announced a new artificial intelligence tool for cloud customers."
}
```

## Notes

- This app is meant for the optional Web Application Frontend bonus.
- It runs locally for free.
- Deployment is optional and can be done later with a free service if required.
