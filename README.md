# AI-Based Classification of Student Substance Use Policies Across Texas Independent School Districts

Description: Using LLMs to classify Texas ISDs policies of student substance use on a spectrum

## Objective

Construct a standardized, quantitative index evaluating official policy documentation across 12 Texas ISDs.

## Overview

Developed a custom web scraping algorithm to collect policy document PDFs from TASB and used DeepSeek-V3.2-Exp to extract frequency of disciplinary and health-related terms that was collectively defined as a team. Afterwards, a quantitative index was calculated to compare each district's approach on a disciplinary-to-supportive spectrum.

## Methods
* Web Scraping
* Natural Language Processing
* LLM Analysis \& Evaluation
* Statistical Analysis

## Languages & Tools

### Programming & Libraries
* Python
* Selenium
* HuggingFace
* torch
* nltk
* Transformers
* Pandas
* NumPy
* Matplotlib
* Seaborn

### LLMs
* DeepSeek-V3.2-Exp
* Llama-4-Scout-17B-16E-Instruct
* Mistral-Large-3-675B-Instruct-2512

## Key Findings
1. Over two-thirds of all coded policy language on student substance use is heavily oriented toward disciplinary terminology rather than health-oriented interventions. The mean disciplinary term ratio overall was 68.9% compared to a mean health term ratio of 31.1%.
2. Districts with higher disciplinary term ratios tend to be located in towns or midsize suburban areas.
3. On the contrast, districts with higher health term ratios tend to be located in rural and small suburban areas.

## Repository Information
* llm-analysis-local.ipynb: This code analyzes the policy PDFs testing different LLMs
* llm-results-analysis.ipynb: Based on the results of the LLM analysis, this code calculates the quantitative index (Soon to be included)
* tasb-web-scraper.ipynb: This is a custom web scraping algorithm using Selenium to scrape the TASB website to download the policy PDF data.

