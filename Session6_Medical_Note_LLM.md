# Practice Session #6: Clinical Data Analysis Using the OpenAI API (Nov 25, 2025)

In this session, we will learn how to perform **clinical data analysis using the OpenAI API**.  

You can find **all Session 6 materials** in the folder below:
    [Session6 Folder](https://github.com/leelabsg/GCDA_tutorial_2025_Fall/tree/main/Session6)

## Dataset Description

This tutorial uses **three types of data**:

- **Shorthand Clinical Notes**  
  Compact notes containing abbreviations and short phrases commonly used in real-world clinical settings.

- **Narrative (MIMIC-style) Clinical Notes**  
  Long-form descriptive notes similar to those from the MIMIC-III/IV dataset (e.g., discharge summaries, progress notes).

- **InBody Scan Images**  
  Body composition analysis images (e.g., weight, skeletal muscle mass, body fat).  
  These are used for multimodal image–text analysis examples.

> **📌 Note**
> This hands-on session uses:
> - **Shorthand clinical notes**  
> - **InBody images**
>
> The **narrative-style clinical notes are not used** in this session.

## Environment Setup

This tutorial assumes:

- You are running the notebook **locally**.
- **Python is already installed** (Python 3.9+ recommended).
- A virtual environment is recommended but optional.
- You already have a **personal OpenAI API key** available for use.

## Required Python Packages

The tutorial uses the following imports:

```python
import pandas as pd
from openai import OpenAI
import requests
import os
import base64
import re
from dotenv import load_dotenv
```

#### Install Required Packages
```
pip install pandas openai python-dotenv
```

## Environment Variables
create a **.env** file in the project directory:
```
API_KEY=your_api_key_here
```
Load it in your notebook or script:
```
load_dotenv()
api_key = os.getenv("API_KEY")
```