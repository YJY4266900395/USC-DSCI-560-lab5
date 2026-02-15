# USC-DSCI-560-lab5
1. Project Overview

This project implements an automated Reddit semantic analysis system that collects, preprocesses, stores, embeds, clusters, and queries Reddit posts from multiple technology-related subreddits.

The system performs:

Large-scale Reddit scraping with rate-limit handling

Data preprocessing and normalization

MySQL database storage with upsert logic

Semantic embedding using SentenceTransformer (all-MiniLM-L6-v2)

K-Means clustering for topic grouping

Interactive query interface

Periodic automation pipeline

The entire workflow runs either independently or as a scheduled automation system.

2. System Architecture

The pipeline consists of five major stages:

Scraping

Database Ingestion

Semantic Embedding

Clustering

Automation & Query

Full workflow:

Scraping → Ingestion → Embedding → Clustering → Database Update → Query/Visualization
3. Database Setup
 MySQL Setup
Ensure MySQL is running locally.
Upsert logic ensures:

No duplicates

Safe re-ingestion

Updated ingestion timestamps
4. Output Artifacts
Core Outputs

centroids.npy

labels.npy

clusters_posts.csv

clusters_summary.json

meta.json

fullnames.json

Visualization

PCA 2D projection

Query cluster highlight plots
