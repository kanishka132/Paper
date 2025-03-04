from flask import Flask, render_template, request, redirect, url_for
import requests
import feedparser
import re
import os
from werkzeug.utils import secure_filename
from transformers import pipeline
from keybert import KeyBERT
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from extraction import extract_title_from_pdf, extract_abstract_keywords_from_pdf, extract_references_from_pdf
from io import BytesIO
import tempfile  
import json
import arxiv
from transformers import pipeline
from keybert import KeyBERT
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Fetch papers from arXiv
def fetch_arxiv_papers(query, max_results=10):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        feed = feedparser.parse(response.text)
        papers = [
            {
                "id": idx,
                "title": preprocess_latex(entry.title),
                "authors": ", ".join([author.name for author in entry.authors]),
                "abstract": preprocess_latex(entry.summary),
                "link": entry.link,
                "pdf_link": entry.link.replace("abs", "pdf"),  # Direct PDF link
                "year": entry.updated[:4],
            }
            for idx, entry in enumerate(feed.entries)
        ]
        return papers
    return []

# Function to preprocess LaTeX expressions
def preprocess_latex(content):
    content = re.sub(r'\$(.*?)\$', r'\\(\1\\)', content)
    content = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', content)
    return content

def download_pdf_from_link(pdf_link, save_dir="downloads"):
    """
    Download a PDF file from a given link and save it locally.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    
    response = requests.get(pdf_link)
    if response.status_code == 200:
        # Create a safe filename from the PDF link
        pdf_filename = pdf_link.split("/")[-1] + ".pdf"
        file_path = os.path.join(save_dir, pdf_filename)
        
        # Write the content to the file
        with open(file_path, "wb") as pdf_file:
            pdf_file.write(response.content)
        
        print(f"PDF saved to {file_path}")
        return file_path
    else:
        raise ValueError(f"Failed to download PDF from {pdf_link}. HTTP Status: {response.status_code}")


# Initialize models
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
kw_model = KeyBERT()
extractive_summarizer = LsaSummarizer()

def improved_summarize_text(text, max_length=130, min_length=30, num_sentences=3):
    try:
        # Abstractive summarization
        abstract_summary = abstractive_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        
        # Extractive summarization
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        extract_summary = extractive_summarizer(parser.document, num_sentences)
        extract_summary = ' '.join([str(sentence) for sentence in extract_summary])
        
        # Combine summaries
        combined_summary = abstract_summary + " " + extract_summary
        
        # Extract keywords
        keywords = kw_model.extract_keywords(combined_summary, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        
        # Generate key points
        key_points = []
        for sentence in nltk.sent_tokenize(combined_summary):
            for keyword, _ in keywords:
                if keyword.lower() in sentence.lower():
                    key_points.append(sentence)
                    break
        
        # Ensure we have at least 3 key points
        while len(key_points) < 3 and len(nltk.sent_tokenize(combined_summary)) > len(key_points):
            for sentence in nltk.sent_tokenize(combined_summary):
                if sentence not in key_points:
                    key_points.append(sentence)
                    break
        
        return key_points[:3]  # Return top 3 key points
    except Exception as e:
        return [f"Error in summarization: {str(e)}"]


# Function to extract keywords
def extract_keywords(text, num_keywords=5):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
    return [keyword[0] for keyword in keywords]


# Function to recommend similar papers based on abstract and keywords similarity
def recommend_papers(target_abstract, target_keywords, papers, top_n=3):
    # Merge abstract and keywords for similarity calculation
    target_text = target_abstract + " " + " ".join(target_keywords)
    
    # Create the vectorizer and merge all abstracts and keywords of the papers
    abstracts_keywords = [paper['abstract'] + " " + " ".join(paper.get('keywords', [])) for paper in papers]
    abstracts_keywords.append(target_text)  # Add target paper to the end

    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(abstracts_keywords)

    # Compute cosine similarity between the target paper and all others
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    # Get indices of top N+1 similar papers (to account for the target paper)
    similar_indices = similarity.argsort()[-top_n-1:][::-1]

    # Get the recommended papers, excluding the target paper
    recommendations = []
    for i in similar_indices:
        if papers[i]['abstract'] != target_abstract:  # Exclude the target paper
            recommendations.append(papers[i])
        if len(recommendations) == top_n:
            break

    # If there aren't enough recommendations, provide a warning
    if len(recommendations) < top_n:
        print(f"Warning: Only {len(recommendations)} similar papers found instead of the requested {top_n}.")
    
    return recommendations

# Function for citation prediction (simple heuristic model)
def predict_citation_impact(abstract, num_authors, num_keywords):
    abstract_length = len(abstract.split())
    return int(abstract_length * 0.1 + num_authors * 5 + num_keywords * 2)


# Function to generate a word cloud
def generate_wordcloud(keywords):
    text = " ".join(keywords)
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="viridis"
    ).generate(text)

    # Save the word cloud image to a static directory
    filepath = os.path.join('static', 'wordcloud.png')
    wordcloud.to_file(filepath)
    return filepath

# Function to generate author heatmap (productivity)
def generate_author_heatmap(papers):
    # Extract authors from papers
    authors = []
    for paper in papers:
        authors.extend(paper['authors'].split(", "))
    
    # Count the number of papers per author
    author_counts = Counter(authors)
    
    # Create a DataFrame for better control and Plotly integration
    author_data = [{"author": author, "papers": count} for author, count in author_counts.items()]
    
    # Create an interactive bar chart using Plotly
    fig = px.bar(
        author_data,
        x='author',
        y='papers',
        # title="Author Productivity",
        labels={'author': 'Author', 'papers': 'Number of Papers'},
        hover_data={'author': True, 'papers': True},  # Tooltip with author and paper count
        color='papers',  # Color the bars based on number of papers
        color_continuous_scale='Magma',  # Choose a color scale
        text='papers',  # Display number of papers on the bar
        template='ggplot2',  
    )

    # Add sorting by the number of papers (descending order)
    fig.update_layout(
        xaxis={'categoryorder': 'total descending'},
        margin={'t': 40, 'b': 100},  # Adjust margins for better readability
        hoverlabel={'bgcolor': 'white', 'font': {'color': 'black'}},  # Style tooltips
        showlegend=False,  # Hide the legend as it's not needed
    )

    # Enable zoom and pan functionalities in the chart
    fig.update_xaxes(tickangle=45)  # Rotate x-axis labels for better readability

    # Save the plot as an HTML file and return the path
    plot_filepath = os.path.join('static', 'author_heatmap.html')
    fig.write_html(plot_filepath)

    return plot_filepath

# fetch keyword trends
def fetch_trends_for_keywords(keywords):
    """
    Function to fetch trends (publication year counts) for a list of keywords.
    
    Parameters:
    - keywords: List of keywords to search papers for.
    
    Returns:
    - trends_data: List of dictionaries containing trends for each keyword.
    """
    trends_data = []
    for keyword in keywords:
        trend_papers = fetch_arxiv_papers(keyword, max_results=50)  # Fetch up to 50 papers for the keyword
        year_counts = Counter([tp['year'] for tp in trend_papers])
        trends_data.append({
            "keyword": keyword,
            "year_counts": [{"Year": year, "Count": count} for year, count in year_counts.items()]
        })
    return trends_data

def map_keywords_to_references(keywords, references):
    """Map keywords to references based on their appearance in reference titles or content."""
    keyword_reference_map = {}
    for keyword in keywords:
        matched_references = [
            reference for reference in references if keyword.lower() in reference.lower()
        ]
        if matched_references:  # Only add the keyword if there are matching references
            keyword_reference_map[keyword] = matched_references
    return keyword_reference_map



# Route: Home
@app.route('/')
def home():
    return render_template('search.html')


# Route: Search
@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').lower()
    num_results = int(request.form.get('num_results', 10))
    papers = fetch_arxiv_papers(query, max_results=num_results)

    # Collect all keywords from the papers
    all_keywords = []
    for paper in papers:
        paper['keywords'] = extract_keywords(paper['abstract'])  # Extract keywords
        all_keywords.extend(paper['keywords'])

    # Generate the word cloud
    wordcloud_image = generate_wordcloud(all_keywords)

    # Count papers by year
    year_counts = Counter([paper['year'] for paper in papers])
    chart_data = [{"Papers": count, "Year": year} for year, count in year_counts.items()]

    # Sort chart_data by year in descending order
    chart_data = sorted(chart_data, key=lambda x: x["Year"], reverse=True)

    # Generate author heatmap
    author_heatmap_image = generate_author_heatmap(papers)

    return render_template(
        'search_results.html',
        query=query,
        papers=papers,
        chart_data=chart_data,
        wordcloud_image=wordcloud_image,
        author_heatmap=author_heatmap_image  # Pass the generated heatmap image
    )


# Route: Paper Details
@app.route('/details/<int:paper_id>')
def details(paper_id):
    query = request.args.get('query', '').lower()
    papers = fetch_arxiv_papers(query)
    paper = papers[paper_id]

    # Add keyword extraction for the selected paper
    paper['keywords'] = extract_keywords(paper['abstract'])  # Ensure keywords are added

    # Extract arXiv ID from the link
    paper_link = paper['link']  # e.g., "https://arxiv.org/abs/2105.00075"
    arxiv_id = paper_link.split("/")[-1]

    pdf_path = download_pdf_from_link(paper['pdf_link'])
    references = extract_references_from_pdf(pdf_path)

    # Summarize the abstract
    summary = improved_summarize_text(paper['abstract'])

    # Fetch trends for each keyword of the paper by calling the external function
    trends_data = fetch_trends_for_keywords(paper['keywords'])

    # Add paper recommendations
    recommendations = recommend_papers(paper['abstract'], paper['keywords'], papers, top_n=10)

    # Add citation prediction
    citation_score = predict_citation_impact(paper['abstract'], len(paper['authors'].split(',')), len(paper['keywords']))
    paper['citation_score'] = citation_score

    keyword_reference_map = map_keywords_to_references(paper['keywords'], references)

    return render_template(
        'paper_details.html',
        paper=paper,
        summary=summary,
        references=references,
        trends_data=trends_data,
        recommendations=recommendations,
        citation_score=citation_score,
        keyword_reference_map = keyword_reference_map
        
    )

    

# Route: Upload PDF
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')

        if file and allowed_file(file.filename):
            # Secure the filename and save it
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Use extraction.py functions to extract metadata
            title = extract_title_from_pdf(filepath)
            extracted_data = extract_abstract_keywords_from_pdf(filepath)
            references = extract_references_from_pdf(filepath)

            # Prepare data to pass to the template
            metadata = {
                "title": title if title else "Title not found",
                "abstract": extracted_data.get("Abstract", "Abstract not found"),
                "keywords": extracted_data.get("Keywords", "Keywords not found"),
                "references": references if references else ["No references found"]
            }
            # Generate the word cloud
            keywords = metadata["keywords"]
            if isinstance(keywords, str):
                keywords = keywords.split(',')  # Split if it's a comma-separated string
                
            wordcloud_path = generate_wordcloud(keywords)

            # Fetch trends for each keyword of the paper
            trends_data = fetch_trends_for_keywords(keywords)

            # Summarize the abstract
            summary = improved_summarize_text(metadata['abstract'])
            
            # Map keywords to references
            keyword_reference_map = map_keywords_to_references(keywords, references)


            # Render results page with extracted metadata and word cloud
            return render_template('upload_result.html', metadata=metadata, summary=summary, 
            wordcloud_image=wordcloud_path, trends_data=trends_data, keyword_reference_map=keyword_reference_map
            )

    return render_template('upload.html')




if __name__ == '__main__':
    app.run(debug=True)
