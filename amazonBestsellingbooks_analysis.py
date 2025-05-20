
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from pathlib import Path

# Set the visualization style with a style that works across matplotlib versions
# Instead of 'seaborn', use a built-in style that's guaranteed to exist
try:
    plt.style.use('ggplot')  # A commonly available style
except Exception as e:
    print(f"Warning: Could not set style: {e}")
    print("Continuing with default matplotlib style")

print(f'Starting analysis with Seaborn version: {sns.__version__}')

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

# Basic data exploration
print('\n--- DATA OVERVIEW ---')
print(f"Time period: {df['Year'].min()} to {df['Year'].max()}")
print(f"Number of books: {len(df)}")
print(f"Number of unique books: {df['Name'].nunique()}")
print(f"Number of authors: {df['Author'].nunique()}")

print("\nFirst 3 rows of data:")
print(df.head(3))

# Create output directory
output_dir = ensure_output_dir()

# Set colors for visualization
fiction_color = '#E63946'     # Red for fiction
nonfiction_color = '#1D3557'  # Blue for non-fiction
genre_colors = [nonfiction_color, fiction_color]

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
print(f"Saved figure to {output_dir / 'genre_distribution.png'}")
plt.close()

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

# Add legend outside the subplots
fig.legend(
    genre_counts.index,
    loc='center right',
    fontsize=12,
    title='Genre'
)

plt.tight_layout()
plt.savefig(output_dir / 'genre_by_year.png', bbox_inches='tight', dpi=300)
print(f"Saved figure to {output_dir / 'genre_by_year.png'}")
plt.close()

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
    
    # Plot Fiction authors if available
    if best_f_authors is not None:
        axes[1].barh(
            y=best_f_authors.index,
            width=best_f_authors.values,
            color=fiction_color,
            edgecolor='black',
            alpha=0.8
        )
        axes[1].set_title('Top Fiction Authors', fontsize=14)
        axes[1].set_xlabel('Number of Bestsellers')
        axes[1].set_yticks(range(len(best_f_authors)))
        axes[1].set_yticklabels(best_f_authors.index, fontsize=10)
        
        # Add value labels
        for i, v in enumerate(best_f_authors.values):
            axes[1].text(v + 0.1, i, str(v), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_authors_by_genre.png', bbox_inches='tight', dpi=300)
    print(f"Saved figure to {output_dir / 'top_authors_by_genre.png'}")
    plt.close()
else:
    print("Warning: Not enough data to create top authors by genre visualization")

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

# Plot 2: Number of unique books
book_counts = []
for author in top_authors.index:
    book_counts.append(len(unique_books[unique_books['Author'] == author]))

axes[1].hlines(
    y=top_authors.index,
    xmin=0,
    xmax=book_counts,
    color=colors,
    linestyles='dashed',
    linewidth=2
)
axes[1].plot(
    book_counts,
    top_authors.index,
    'o',
    markersize=8,
    color='black'
)
axes[1].set_xlabel('Number of Unique Books')
axes[1].set_title('Unique Bestselling Books', fontsize=12, fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.7)

# Plot 3: Total reviews
review_counts = []
for author in top_authors.index:
    total_reviews = unique_books[unique_books['Author'] == author]['Reviews'].sum() / 1000
    review_counts.append(total_reviews)

bars = axes[2].barh(
    y=top_authors.index,
    width=review_counts,
    color=colors,
    edgecolor='black',
    height=0.6
)

# Add value labels to bars
for i, (bar, value) in enumerate(zip(bars, review_counts)):
    axes[2].text(
        value + 0.5,
        i,
        f'{value:.1f}K',
        va='center',
        fontweight='bold'
    )

axes[2].set_xlabel('Total Reviews (thousands)')
axes[2].set_title('Total Reader Reviews', fontsize=12, fontweight='bold')
axes[2].grid(True, linestyle='--', alpha=0.7)

# Set common y-axis label and ticks
fig.suptitle('Top 20 Amazon Bestselling Authors (2009-2019)', fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.savefig(output_dir / 'top_20_authors_analysis.png', bbox_inches='tight', dpi=300)
print(f"Saved figure to {output_dir / 'top_20_authors_analysis.png'}")
plt.close()

print('\n--- ANALYSIS COMPLETE ---')
print(f"All visualizations have been saved to the '{output_dir}' directory")
print(f"NOTE: You can find the visualizations in: {output_dir.absolute()}")

import pandas as pd # dataframe manipulation
import numpy as np # linear algebra

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
print('Seaborn verion', sns.__version__)
sns.set_style('whitegrid')

# text data
import string
df = pd.read_csv('bestsellers with categories.csv')


df.rename(columns={"User Rating": "User_Rating"}, inplace=True)
df[df.Author == 'J. K. Rowling']
df[df.Author == 'J.K. Rowling']
df.loc[df.Author == 'J. K. Rowling', 'Author'] = 'J.K. Rowling'
df['name_len'] = df['Name'].apply(lambda x: len(x) - x.count(" ")) # subtract whitespaces
punctuations = string.punctuation
print('list of punctuations : ', punctuations)

# percentage of punctuations
def count_punc(text):
    """This function counts the number of punctuations in a text"""
    count = sum(1 for char in text if char in punctuations)
    return round(count/(len(text) - text.count(" "))*100, 3)

# apply function
df['punc%'] = df['Name'].apply(lambda x: count_punc(x))


no_dup = df.drop_duplicates('Name')
g_count = no_dup['Genre'].value_counts()

fig, ax = plt.subplots(figsize=(8, 8))

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%\n({v:d})'.format(p=pct,v=val)
    return my_autopct

genre_col = ['navy','crimson']
#genre_col = ['khaki','plum']

center_circle = plt.Circle((0, 0), 0.7, color='white')
plt.pie(x=g_count.values, labels=g_count.index, autopct=make_autopct(g_count.values), 
          startangle=90, textprops={'size': 15}, pctdistance=0.5, colors=genre_col)
ax.add_artist(center_circle)

fig.suptitle('Distribution of Genre for all unique books from 2009 to 2019', fontsize=20)
fig.show()



y1 = np.arange(2009, 2014)
y2 = np.arange(2014, 2020)
g_count = df['Genre'].value_counts()

fig, ax = plt.subplots(2, 6, figsize=(12,6))

ax[0,0].pie(x=g_count.values, labels=None, autopct='%1.1f%%',
            startangle=90, textprops={'size': 12, 'color': 'white'},
            pctdistance=0.5, radius=1.3, colors=genre_col)
ax[0,0].set_title('2009 - 2019\n(Overall)', color='darkgreen', fontdict={'fontsize': 15})

for i, year in enumerate(y1):
    counts = df[df['Year'] == year]['Genre'].value_counts()
    ax[0,i+1].set_title(year, color='darkred', fontdict={'fontsize': 15})
    ax[0,i+1].pie(x=counts.values, labels=None, autopct='%1.1f%%', 
                  startangle=90, textprops={'size': 12,'color': 'white'}, 
                  pctdistance=0.5, colors=genre_col, radius=1.1)

for i, year in enumerate(y2):
    counts = df[df['Year'] == year]['Genre'].value_counts()
    ax[1,i].pie(x=counts.values, labels=None, autopct='%1.1f%%', 
                startangle=90, textprops={'size': 12,'color': 'white'},
                pctdistance=0.5, colors=genre_col, radius=1.1)
    ax[1,i].set_title(year, color='darkred', fontdict={'fontsize': 15})

#plt.suptitle('Distribution of Fiction and Non-Fiction books for every year from 2009 to 2019',
             #fontsize=25)
fig.legend(g_count.index, loc='center right', fontsize=12)
fig.show()


best_nf_authors = df.groupby(['Author', 'Genre']).agg({'Name': 'count'}).unstack()['Name', 'Non Fiction'].sort_values(ascending=False)[:11]
best_f_authors = df.groupby(['Author', 'Genre']).agg({'Name': 'count'}).unstack()['Name', 'Fiction'].sort_values(ascending=False)[:11]

with plt.style.context('Solarize_Light2'):
    fig, ax = plt.subplots(1, 2, figsize=(8,8))
    
    ax[0].barh(y=best_nf_authors.index, width=best_nf_authors.values,
           color=genre_col[0])
    ax[0].invert_xaxis()
    ax[0].yaxis.tick_left()
    ax[0].set_xticks(np.arange(max(best_f_authors.values)+1))
    ax[0].set_yticklabels(best_nf_authors.index, fontsize=12, fontweight='semibold')
    ax[0].set_xlabel('Number of appreances')
    ax[0].set_title('Non Fiction Authors')
    
    ax[1].barh(y=best_f_authors.index, width=best_f_authors.values,
           color=genre_col[1])
    ax[1].yaxis.tick_right()
    ax[1].set_xticks(np.arange(max(best_f_authors.values)+1))
    ax[1].set_yticklabels(best_f_authors.index, fontsize=12, fontweight='semibold')
    ax[1].set_title('Fiction Authors')
    ax[1].set_xlabel('Number of appreances')
    
    fig.legend(['Non Fiction', 'Fiction'], fontsize=12)
    plt.show()


n_best = 20

top_authors = df.Author.value_counts().nlargest(n_best)
no_dup = df.drop_duplicates('Name') # removes all rows with duplicate book names

fig, ax = plt.subplots(1, 3, figsize=(11,10), sharey=True)

color = sns.color_palette("hls", n_best)

ax[0].hlines(y=top_authors.index , xmin=0, xmax=top_authors.values, color=color, linestyles='dashed')
ax[0].plot(top_authors.values, top_authors.index, 'go', markersize=9)
ax[0].set_xlabel('Number of appearences')
ax[0].set_xticks(np.arange(top_authors.values.max()+1))
ax[0].set_yticklabels(top_authors.index, fontweight='semibold')
ax[0].set_title('Appearences')

book_count = []
total_reviews = []
for name, col in zip(top_authors.index, color):
    book_count.append(len(no_dup[no_dup.Author == name]['Name']))
    total_reviews.append(no_dup[no_dup.Author == name]['Reviews'].sum()/1000)
ax[1].hlines(y=top_authors.index , xmin=0, xmax=book_count, color=color, linestyles='dashed')
ax[1].plot(book_count, top_authors.index, 'go', markersize=9)
ax[1].set_xlabel('Number of unique books')
ax[1].set_xticks(np.arange(max(book_count)+1))
ax[1].set_title('Unique books')

ax[2].barh(y=top_authors.index, width=total_reviews, color=color, edgecolor='black', height=0.7)
for name, val in zip(top_authors.index, total_reviews):
    ax[2].text(val+2, name, val)
ax[2].set_xlabel("Total Reviews (in 1000's)")
ax[2].set_title('Total reviews')

#plt.suptitle('Top 20 best selling Authors (from 2009 to 2019) details', fontsize=15)
plt.show()



