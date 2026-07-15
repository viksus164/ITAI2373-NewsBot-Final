# ITAI 2373 NewsBot Intelligence System 2.0

## Project Overview

This repository contains my ITAI 2373 final project, NewsBot Intelligence System 2.0. The project builds on my midterm NewsBot and expands it into a more complete NLP platform for news analysis.

NewsBot 2.0 can classify articles, summarize content, analyze sentiment, extract named entities, discover hidden topics, find similar articles, process simple multilingual examples, answer basic natural-language queries, and run through a Gradio web application inside Google Colab.

The goal of this project is to show how multiple NLP techniques can work together inside one practical system instead of staying as separate notebook exercises.

## Repository Structure

This project uses a Google Colab notebook-centered structure. The final implementation is contained in one complete notebook because the project was developed and tested in Colab.

```text
ITAI2373-NewsBot-Final/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- NewsBot2_Final_Project_Viktoriya_Kurmisheva.ipynb
|-- notebooks/
|   `-- NewsBot2_Final_Project_Viktoriya_Kurmisheva.ipynb
|-- reports/
|   |-- FP_TechnicalDoc_ViktoriyaKurmisheva_TeamTesters_ITAI2373.pdf
|   |-- FP_ExecutiveSummary_ViktoriyaKurmisheva_TeamTesters_ITAI2373.pdf
|   |-- FP_ReflectiveJournal_TeamTesters_ITAI2373.pdf
|   `-- FP_Presentation_ViktoriyaKurmisheva_TeamTesters_ITAI2373.pptx
|-- docs/
|   |-- FP_TechnicalDoc_ViktoriyaKurmisheva_TeamTesters_ITAI2373.docx
|   |-- FP_ExecutiveSummary_ViktoriyaKurmisheva_TeamTesters_ITAI2373.docx
|   `-- FP_ReflectiveJournal_TeamTesters_ITAI2373.docx
`-- web_app/
    |-- README.md
    |-- app.py
    |-- newsbot_engine.py
    |-- templates/
    `-- static/
```

The notebook includes the same major modules described in the final project instructions: data processing, enhanced classification, topic modeling, language understanding and generation, multilingual analysis, conversational interface, system integration, testing, evaluation, and a web application frontend.

## Main Features

- Loads and validates the BBC News Classification dataset.
- Cleans and preprocesses article text.
- Trains an enhanced news classifier with confidence scoring.
- Compares multiple machine learning models.
- Uses TF-IDF features, sentiment features, and article length features.
- Discovers hidden topics with LDA and NMF.
- Tracks sentiment patterns by category and generated analysis month.
- Extracts named entities with spaCy.
- Builds entity co-occurrence relationships.
- Creates extractive article summaries without paid APIs.
- Finds similar articles with semantic search.
- Demonstrates multilingual analysis with language detection and manual/offline translation examples.
- Provides a simple conversational interface for user queries.
- Includes a Gradio web application frontend in Colab.
- Includes an advanced research extension for bias/framing analysis.

## Dataset

The project uses the BBC News Classification dataset from Kaggle. The dataset ZIP file is not included in this repository. When running the notebook in Colab, upload `learn-ai-bbc.zip` when the dataset cell asks for it.

## How to Run the Project in Google Colab

1. Open `NewsBot2_Final_Project_Viktoriya_Kurmisheva.ipynb` in Google Colab.
2. Run the setup/import cells from the top.
3. Download the required (learn-ai-bbc.zip) here: https://www.kaggle.com/competitions/learn-ai-bbc/data
4. When prompted, upload `learn-ai-bbc.zip`. 
5. Continue running the notebook cells in order.
6. Near the end, run the Gradio web app cell to test the interactive dashboard.

No paid API keys are required.

## Team Testers Peer Review

This was not one shared notebook project. Abraham Barreto and I worked as peer testers. Each of us built an individual NewsBot system, then reviewed and tested the other person's project for clarity, usability, and quality.

## Important Notes

- Do not upload `kaggle.json`.
- Do not upload `learn-ai-bbc.zip`.
- Do not upload extracted dataset folders such as `bbc_final_data/`. 
- Run the notebook cells in order because later sections depend on variables created earlier.
- The BBC dataset does not include real publication dates, so generated analysis months are used only for trend demonstration.
