# Amazon Bestselling Books Analysis 

## Overview

This document provides comprehensive documentation for the Python script analyzing Amazon's bestselling books from 2009 to 2019. The script processes book data, performs statistical analysis, and generates insightful visualizations to understand trends in the bestseller market.

## Table of Contents

1. [Dataset Information](#dataset-information)
2. [Analysis Methodology](#analysis-methodology)
3. [Preprocessing Steps](#preprocessing-steps)
4. [Visualizations](#visualizations)
   - [Genre Distribution](#visualization-1-genre-distribution)
   - [Genre Distribution by Year](#visualization-2-genre-distribution-by-year)
   - [Top Authors by Genre](#visualization-3-top-authors-by-genre)
   - [Top 20 Authors Analysis](#visualization-4-top-20-authors-analysis)
5. [Code Structure](#code-structure)
6. [Functions Reference](#functions-reference)
7. [Dependencies](#dependencies)
8. [Appendix: Complete Code Explanation](#appendix-complete-code-explanation)

## Dataset Information

The analysis uses the "bestsellers with categories.csv" dataset, which contains information about Amazon's bestselling books from 2009 to 2019. The dataset includes the following fields:

- **Name**: Book title
- **Author**: Book author
- **User Rating**: Average user rating (likely on a scale of 1-5)
- **Reviews**: Number of user reviews
- **Price**: Book price
- **Year**: Year when the book appeared on the bestseller list
- **Genre**: Book genre (Fiction or Non Fiction)

## Analysis Methodology

The analysis employs both descriptive statistics and data visualization to identify patterns and trends in Amazon's bestselling books. Key analytical approaches include:

1. **Temporal Analysis**: Examining how book characteristics and genre distributions change over time (2009-2019)
2. **Author Performance Analysis**: Identifying top-performing authors and analyzing their impact
3. **Genre Analysis**: Comparing fiction vs. non-fiction bestsellers
4. **Text Analysis**: Examining book title characteristics (length, punctuation usage)

## Preprocessing Steps

Before analysis, the script performs several data preprocessing steps:

1. **Column Renaming**: Removes spaces from column names for easier access
   ```python
   df.rename(columns={"User Rating": "User_Rating"}, inplace=True)
   ```

2. **Author Name Standardization**: Fixes inconsistencies in author names
   ```python
   df.loc[df.Author == 'J. K. Rowling', 'Author'] = 'J.K. Rowling'
   ```

3. **Feature Engineering**: Creates new features for text analysis
   ```python
   # Calculate book name length (excluding spaces)
   df['name_len'] = df['Name'].apply(lambda x: len(x) - x.count(" "))
   # Calculate punctuation percentage
   df['punc%'] = df['Name'].apply(count_punctuation)
   ```

## Visualizations

The script generates four key visualizations to present the analysis results.

### Visualization 1: Genre Distribution

A pie chart showing the overall distribution of fiction versus non-fiction books among Amazon's bestsellers from 2009-2019.

**Implementation Details:**
- Uses unique books only (removes duplicates) to avoid counting the same book multiple times
- Implements a custom percentage labeling function to show both percentages and counts
- Uses distinct colors for fiction (red) and non-fiction (blue)
- Adds a white center circle for better visual appearance

**Visual Insights:**
- Shows the relative proportion of fiction versus non-fiction books in the bestseller list
- Provides exact counts and percentages for each genre

### Visualization 2: Genre Distribution by Year

A collection of pie charts showing how the genre distribution has evolved year by year from 2009 to 2019.

**Implementation Details:**
- First subplot shows the overall distribution across all years
- Subsequent subplots show distribution for individual years
- Uses consistent color scheme with the first visualization
- Handles edge cases where a year might have only one genre present

**Visual Insights:**
- Reveals temporal trends in genre popularity
- Shows how consumer preferences may have shifted over the decade
- Enables year-to-year comparison of fiction vs. non-fiction popularity

### Visualization 3: Top Authors by Genre

A horizontal bar chart comparing the top 10 bestselling authors in fiction and non-fiction categories.

**Implementation Details:**
- Uses a split design with non-fiction authors on the left and fiction authors on the right
- Implements robust error handling for cases where data might be missing
- Uses consistent color coding with previous visualizations
- Displays the exact number of bestsellers for each author

**Visual Insights:**
- Identifies the most successful authors in each genre
- Allows comparison of author dominance between fiction and non-fiction
- Shows the competitiveness of each genre's author landscape

### Visualization 4: Top 20 Authors Analysis

A three-panel visualization analyzing the top 20 bestselling authors across multiple dimensions.

**Implementation Details:**
- Panel 1: Number of appearances in the bestseller list
- Panel 2: Number of unique bestselling books
- Panel 3: Total reader reviews (in thousands)
- Uses dumbbell plots for the first two panels and a bar chart for the third
- Employs a consistent color palette across all three panels

**Visual Insights:**
- Compares author success across different metrics
- Reveals authors who have multiple bestselling books versus those who have fewer books with longer bestseller list presence
- Shows reader engagement (through review counts) for top authors

## Code Structure

The script is organized into several logical sections:

1. **Imports and Setup**: Imports necessary libraries and configures visualization settings
2. **Utility Functions**: Defines helper functions for analysis and visualization
3. **Data Loading and Preprocessing**: Loads the dataset and performs initial cleaning
4. **Data Exploration**: Provides basic dataset statistics
5. **Visualization Sections**: Four distinct sections, each creating one visualization
6. **Output Management**: Creates output directory and saves visualizations

## Functions Reference

### `count_punctuation(text)`

Calculates the percentage of punctuation characters in a text string.

**Parameters:**
- `text` (str): The input text string

**Returns:**
- `float`: Percentage of punctuation characters (rounded to 3 decimal places)

### `make_autopct(values)`

Creates a custom function for percentage labeling in pie charts.

**Parameters:**
- `values` (list): List of values used in the pie chart

**Returns:**
- `function`: A function that formats percentage labels with both percentage and count

### `ensure_output_dir()`

Creates a directory for saving visualization outputs if it doesn't exist.

**Parameters:**
- None

**Returns:**
- `Path`: Path object pointing to the output directory

## Dependencies

The script relies on the following Python libraries:

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **matplotlib**: Creating visualizations
- **seaborn**: Enhanced visualization aesthetics
- **string**: String manipulation utilities (for punctuation analysis)
- **pathlib**: Path handling for cross-platform compatibility

## Appendix: Complete Code Explanation

### Setting Up the Environment

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from pathlib import Path

# Set the visualization style with a style that works across matplotlib versions
try:
    plt.style.use('ggplot')  # A commonly available style
except Exception as e:
    print(f"Warning: Could not set style: {e}")
    print("Continuing with default matplotlib style")

print(f'Starting analysis with Seaborn version: {sns.__version__}')
```

This section imports the necessary libraries and configures the visualization environment. The code includes robust error handling to ensure it works across different matplotlib versions.

### Utility Functions

```python
# Function to calculate percentage of punctuation in text
def count_punctuation(text):
    """Calculate the percentage of punctuation in a text"""
    punctuations = string.punctuation
    count = sum(1 for char in text if char in punctuations)
    return round(count/(len(text) - text.count(" "))*100, 3)

# Function to create custom percentage labels for pie charts
def make_autopct(values):
    """Create custom percentage labels that include counts"""
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)
    return my_autopct

# Function to ensure output directory exists
def ensure_output_dir():
    """Create directory for saving figures if it doesn't exist"""
    output_dir = Path('amazon_book_analysis')
    output_dir.mkdir(exist_ok=True)
    return output_dir
```

These utility functions help with text analysis and visualization formatting:
- `count_punctuation`: Analyzes text characteristics
- `make_autopct`: Enhances pie chart readability
- `ensure_output_dir`: Manages file output organization

### Data Loading and Preprocessing

```python
# Load the dataset
print('\n--- LOADING DATA ---')
file_path = 'bestsellers with categories.csv'
df = pd.read_csv(file_path)
print(f"Successfully loaded data with {df.shape[0]} books")

# Basic data preprocessing
print('\n--- PREPROCESSING DATA ---')
# Rename column to remove space
df.rename(columns={"User Rating": "User_Rating"}, inplace=True)

# Fix author name inconsistency
print("Fixing author name inconsistencies...")
df.loc[df.Author == 'J. K. Rowling', 'Author'] = 'J.K. Rowling'

# Add features for text analysis
print("Adding text analysis features...")
# Calculate book name length (excluding spaces)
df['name_len'] = df['Name'].apply(lambda x: len(x) - x.count(" "))
# Calculate punctuation percentage
df['punc%'] = df['Name'].apply(count_punctuation)
```

This section loads the dataset and performs initial preprocessing:
- Loads the CSV file into a pandas DataFrame
- Renames columns to follow consistent naming conventions
- Fixes inconsistencies in author names
- Creates derived features for text analysis

### Data Exploration

```python
# Basic data exploration
print('\n--- DATA OVERVIEW ---')
print(f"Time period: {df['Year'].min()} to {df['Year'].max()}")
print(f"Number of books: {len(df)}")
print(f"Number of unique books: {df['Name'].nunique()}")
print(f"Number of authors: {df['Author'].nunique()}")

print("\nFirst 3 rows of data:")
print(df.head(3))
```

This section provides basic statistics about the dataset:
- Time range covered
- Total number of book entries
- Number of unique books (excluding duplicates)
- Number of unique authors
- Preview of the first few rows

### Visualization 1: Genre Distribution

```python
# VISUALIZATION 1: Overall Genre Distribution Pie Chart
print('\n--- CREATING GENRE DISTRIBUTION PIE CHART ---')
# Remove duplicate books to avoid counting the same book multiple times
unique_books = df.drop_duplicates('Name')
genre_counts = unique_books['Genre'].value_counts()

plt.figure(figsize=(10, 8))
ax = plt.subplot(111)

# Create pie chart with a white circle in the middle for better appearance
center_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(
    x=genre_counts.values, 
    labels=genre_counts.index, 
    autopct=make_autopct(genre_counts.values),
    startangle=90, 
    textprops={'size': 15}, 
    pctdistance=0.5, 
    colors=genre_colors
)
ax.add_artist(center_circle)

plt.title('Distribution of Genres for Amazon Bestsellers (2009-2019)', fontsize=16)
plt.savefig(output_dir / 'genre_distribution.png', bbox_inches='tight', dpi=300)
```

This code creates a donut chart showing the distribution of fiction versus non-fiction books in the bestseller list.

### Visualization 2: Genre Distribution by Year

```python
# VISUALIZATION 2: Genre Distribution by Year
print('\n--- CREATING GENRE DISTRIBUTION BY YEAR ---')
years = range(2009, 2020)
fig, axes = plt.subplots(2, 6, figsize=(16, 8))
axes = axes.flatten()  # Convert 2D array to 1D for easier indexing

# First subplot shows overall distribution
overall_counts = df['Genre'].value_counts()
axes[0].pie(
    x=overall_counts.values,
    labels=None,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'size': 10, 'color': 'white', 'fontweight': 'bold'},
    pctdistance=0.5,
    colors=genre_colors
)
axes[0].set_title('2009-2019\n(Overall)', fontsize=12, fontweight='bold')

# Create pie chart for each year
for i, year in enumerate(years):
    year_data = df[df['Year'] == year]
    year_counts = year_data['Genre'].value_counts()
    
    # Handle years with only one genre present
    if len(year_counts) == 1:
        missing_genre = set(overall_counts.index) - set(year_counts.index)
        if missing_genre:
            year_counts[list(missing_genre)[0]] = 0
    
    pie_idx = i + 1  # Skip first position (used for overall)
    if pie_idx < len(axes):  # Ensure we don't exceed available subplots
        axes[pie_idx].pie(
            x=year_counts.values,
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'size': 10, 'color': 'white', 'fontweight': 'bold'},
            pctdistance=0.5,
            colors=genre_colors
        )
        axes[pie_idx].set_title(f'{year}', fontsize=12, fontweight='bold')
```

This section creates a multi-panel visualization showing genre distribution trends over time.

### Visualization 3: Top Authors by Genre

```python
# VISUALIZATION 3: Top Authors by Genre
print('\n--- CREATING TOP AUTHORS BY GENRE VISUALIZATION ---')
# Get top authors for non-fiction and fiction
author_genre_counts = df.groupby(['Author', 'Genre']).size().unstack(fill_value=0)

# Sort and get top 10 for each genre - handle cases where columns might not exist
best_nf_authors = None
best_f_authors = None

# Check for Non Fiction column
if 'Non Fiction' in author_genre_counts.columns:
    best_nf_authors = author_genre_counts['Non Fiction'].sort_values(ascending=False).head(10)

# Check for Fiction column
if 'Fiction' in author_genre_counts.columns:
    best_f_authors = author_genre_counts['Fiction'].sort_values(ascending=False).head(10)

# Create figure only if we have data
if best_nf_authors is not None or best_f_authors is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    
    # Plot Non-Fiction authors if available
    if best_nf_authors is not None:
        axes[0].barh(
            y=best_nf_authors.index,
            width=best_nf_authors.values,
            color=nonfiction_color,
            edgecolor='black',
            alpha=0.8
        )
        axes[0].set_title('Top Non-Fiction Authors', fontsize=14)
        axes[0].set_xlabel('Number of Bestsellers')
        axes[0].invert_xaxis()  # Reverse x-axis for Non-Fiction side
        axes[0].set_yticks(range(len(best_nf_authors)))
        axes[0].set_yticklabels(best_nf_authors.index, fontsize=10)
        
        # Add value labels
        for i, v in enumerate(best_nf_authors.values):
            axes[0].text(v - 0.5, i, str(v), va='center', fontweight='bold')
```

This code identifies and visualizes the most successful authors in each genre category.

### Visualization 4: Top 20 Authors Analysis

```python
# VISUALIZATION 4: Top 20 Authors Analysis
print('\n--- CREATING TOP 20 AUTHORS ANALYSIS ---')
# Get top 20 authors by number of appearances
top_authors = df['Author'].value_counts().nlargest(20)
unique_books = df.drop_duplicates('Name')

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=True)

# Generate colors for each author - use a colormap that exists across versions
colors = sns.color_palette("viridis", len(top_authors))

# Plot 1: Number of appearances
axes[0].hlines(
    y=top_authors.index,
    xmin=0,
    xmax=top_authors.values,
    color=colors,
    linestyles='dashed',
    linewidth=2
)
axes[0].plot(
    top_authors.values,
    top_authors.index,
    'o',
    markersize=8,
    color='black'
)
axes[0].set_xlabel('Number of Appearances in Bestseller List')
axes[0].set_title('Total Bestseller Appearances', fontsize=12, fontweight='bold')
axes[0].grid(True, linestyle='--', alpha=0.7)
```

This section creates a detailed multi-metric analysis of the top 20 bestselling authors.

### Output Management

```python
print('\n--- ANALYSIS COMPLETE ---')
print(f"All visualizations have been saved to the '{output_dir}' directory")
print(f"NOTE: You can find the visualizations in: {output_dir.absolute()}")
```

The script concludes by providing the user with information about where the output files are stored.
