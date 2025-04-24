import pandas as pd
import numpy as np
import re
import nltk
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PyPDF2 import PdfReader
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained model and vectorizer
def load_model():
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/vectorization.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Extract the text from uploaded file
def extract_text(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        st.warning("Invalid file format!")
        return ""

# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Predict cluster
def predict_cluster(text, model, vectorizer_dict):
    cleaned = clean_text(text)

    # Extract vectorization components
    tokenizer = vectorizer_dict['tokenizer']
    bert_model = vectorizer_dict['bert_model']
    pca = vectorizer_dict['pca']
    umap = vectorizer_dict['umap']

    # BERT embedding
    inputs = tokenizer(cleaned, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # Apply PCA and UMAP
    reduced = pca.transform([cls_embedding])
    final_vector = umap.transform(reduced)

    # Predict cluster and membership
    probs = model.u[model.predict(final_vector)[0]]
    cluster_id = np.argmax(probs)

    return cluster_id, cleaned, probs, final_vector

def cluster_distribution(probs):
    if isinstance(probs, (np.ndarray, list)):
        st.markdown("### Cluster Membership Probabilities")
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Create a DataFrame for better visualization
        prob_df = pd.DataFrame({
            'Cluster': [f"Cluster {i}" for i in range(len(probs))],
            'Probability (%)': np.array(probs) * 100
        })
        
        # Sort by probability
        prob_df = prob_df.sort_values('Probability (%)', ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(prob_df['Cluster'], prob_df['Probability (%)'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(probs))))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}%', ha='left', va='center')
        
        ax.set_xlim(0, 100)
        ax.set_xlabel('Membership Probability (%)')
        ax.set_title('Document Membership Across Clusters')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
    else:
        st.warning("Could not retrieve cluster probabilities.")

def scatter_plot(final_vector, cluster_id, model):
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all cluster centers from your model
    cluster_centers = model.centers
    for i, center in enumerate(cluster_centers):
        ax.scatter(center[0], center[1], 
                  c=[i], cmap='viridis',
                  s=300, alpha=0.7, 
                  edgecolors='k', linewidth=2,
                  label=f'Cluster {i} Center')
    
    # Plot the new document
    ax.scatter(final_vector[:, 0], final_vector[:, 1], 
              c=[cluster_id], cmap='viridis',
              s=200, alpha=1.0,
              edgecolors='r', linewidth=3,
              marker='*', label='Your Document')
    
    # Add annotations and styling
    ax.set_title('Document Position Relative to Cluster Centers', pad=20)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.grid(alpha=0.2)
    ax.legend(loc='best')
    
    # Add annotation for the current document
    ax.annotate(f'Your Document (Cluster {cluster_id})', 
                xy=(final_vector[0, 0], final_vector[0, 1]),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    st.pyplot(fig)

def display_cluster_info(cluster_id):
    """Display information about what each cluster represents"""
    cluster_descriptions = {
        0: "Climate Change and Policy Discussions",
        1: "Biodiversity and Conservation Efforts",
        2: "Renewable Energy and Technology",
        3: "Environmental Activism and Social Movements",
        4: "Pollution and Environmental Health"
    }
    
    if cluster_id in cluster_descriptions:
        st.markdown(f"### Cluster {cluster_id} Characteristics")
        st.info(f"**Primary Theme:** {cluster_descriptions[cluster_id]}")
        
        # Example keywords (you would replace with actual keywords from your analysis)
        cluster_keywords = {
            0: ["climate change", "carbon emissions", "paris agreement", "global warming"],
            1: ["biodiversity", "wildlife", "conservation", "endangered species"],
            2: ["solar power", "wind energy", "renewables", "clean technology"],
            3: ["protest", "activism", "environmental justice", "grassroots"],
            4: ["air pollution", "plastic waste", "toxic chemicals", "public health"]
        }
        
        st.markdown("**Common Keywords:**")
        st.write(", ".join(cluster_keywords[cluster_id]))
        
        st.markdown("**Typical Articles:**")
        st.write("- News about international climate agreements")
        st.write("- Scientific reports on environmental impacts")
        st.write("- Policy debates on environmental regulations")

def main():
    st.set_page_config(layout="wide", page_title="NLP Clustering of Environmental Discourse", page_icon="🌍")
    
    # Custom CSS styling
    st.markdown("""
    <style>
        .header {
            font-size: 36px !important;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #2e86ab;
        }
        .subheader {
            font-size: 24px !important;
            text-align: center;
            margin-bottom: 30px;
            color: #4a7c59;
        }
        .stButton>button {
            background-color: #4a7c59;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #3a5a40;
            color: white;
        }
        .stTextArea>div>div>textarea {
            border-radius: 5px;
            padding: 1rem;
        }
        .result-box {
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .cluster-0 { background-color: #f8f9fa; }
        .cluster-1 { background-color: #e9f5ff; }
        .cluster-2 { background-color: #e6f9e6; }
        .cluster-3 { background-color: #fff8e6; }
        .cluster-4 { background-color: #ffebee; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="header">Environmental Discourse Clustering</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Analyzing News Articles with NLP and Fuzzy Clustering</p>', unsafe_allow_html=True)
    
    # Introduction
    with st.expander("About this tool"):
        st.write("""
        This application uses advanced Natural Language Processing (NLP) and machine learning techniques 
        to analyze environmental news articles from The Guardian. It automatically categorizes articles 
        into thematic clusters based on their content.
        
        - **How it works**: Upload a text document or paste article text, and the system will:
            1. Process and clean the text
            2. Extract semantic features using BERT embeddings
            3. Reduce dimensionality with PCA and UMAP
            4. Assign to a cluster using Fuzzy C-Means clustering
            
        - **Use cases**: Media analysis, content categorization, trend monitoring in environmental journalism
        """)
    
    # Load model
    model, vectorization = load_model()
    
    # Input section
    st.markdown("## Input Your Document")
    input_method = st.radio("Select input method:", ("Paste text", "Upload file"), horizontal=True)
    
    if input_method == "Paste text":
        text_input = st.text_area("Enter article text here...", height=200,
                                 placeholder="Paste the full text of an environmental news article...")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
        text_input = None
    
    if st.button("Analyze Document"):
        if text_input or uploaded_file:
            with st.spinner("Analyzing document content..."):
                try:
                    raw_text = extract_text(uploaded_file) if uploaded_file else text_input
                    if not raw_text.strip():
                        st.warning("Please provide some text to analyze!")
                        st.stop()
                    
                    # Show processing steps
                    with st.expander("Processing Steps", expanded=True):
                        st.write("1. **Text Cleaning**: Removing special characters, stopwords, and lemmatizing")
                        cleaned_text = clean_text(raw_text)
                        st.write("2. **Feature Extraction**: Generating BERT embeddings (512-dimensional)")
                        st.write("3. **Dimensionality Reduction**: Applying PCA and UMAP for visualization")
                        st.write("4. **Cluster Assignment**: Calculating membership probabilities")
                    
                    cluster_id, cleaned_text, probs, final_vector = predict_cluster(raw_text, model, vectorization)
                    
                    # Display results in a nicely formatted box
                    st.markdown("## Analysis Results")
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f'<div class="result-box cluster-{cluster_id}">', unsafe_allow_html=True)
                            st.metric(label="Primary Cluster Assignment", 
                                    value=f"Cluster {cluster_id}",
                                    help="The cluster with highest membership probability")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show top 3 clusters
                            top_clusters = np.argsort(probs)[-3:][::-1]
                            st.write("**Top 3 Clusters:**")
                            for i, c in enumerate(top_clusters):
                                st.progress(int(probs[c]*100), 
                                f"Cluster {c}: {probs[c]*100:.1f}%")
                                
                        with col2:
                            st.markdown("**Processed Text Sample:**")
                            with st.expander("View processed text"):
                                st.text(cleaned_text[:1000] + ("..." if len(cleaned_text) > 1000 else ""))
                    
                    # Visualizations in tabs
                    tab1, tab2, tab3 = st.tabs(["Cluster Probabilities", "Embedding Visualization", "Cluster Information"])
                    
                    with tab1:
                        cluster_distribution(probs)
                    
                    with tab2:
                        scatter_plot(final_vector, cluster_id, model)
                        st.caption("Note: This shows the document's position in the reduced 2D space relative to the training data clusters.")
                    
                    with tab3:
                        display_cluster_info(cluster_id)
                    
                    # Add download button for results
                    result_str = f"Document Analysis Results\nPrimary Cluster: {cluster_id}\n\nCluster Probabilities:\n"
                    for i, p in enumerate(probs):
                        result_str += f"Cluster {i}: {p*100:.1f}%\n"
                    
                    st.download_button("Download Results as TXT", 
                                     data=result_str,
                                     file_name="cluster_analysis_results.txt")
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    st.exception(e)
        else:
            st.warning("Please provide either text input or upload a file!")
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Environmental Discourse Analysis Tool | Powered by BERT and Fuzzy C-Means Clustering</p>
        <p>For research and educational purposes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()