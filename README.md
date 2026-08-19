1. Main Application Code 
# app.py 
import streamlit as st 
import requests 
import json 
import pandas as pd 
from PIL import Image 
import torch 
import torchvision.transforms as transforms import torchvision.models as models 
import torch.nn as nn 
from torchvision import transforms 
import pickle 
import os 
from datetime import datetime 
import matplotlib.pyplot as plt 
# Set page configuration 
st.set_page_config( 
 page_title="RemedicaAI", 
 page_icon="��", 
 layout="wide" 
) 
# Custom CSS 
st.markdown(""" 
<style> 
 .main-header { 
 font-size: 2.5rem; 
 color: #2E8B57; 
 text-align: center; 
 margin-bottom: 1rem; 
 } 
 .sub-header { 
 font-size: 1.5rem; 
 color: #3CB371; 
 margin-top: 2rem; 
 } 
 .info-box { 
 background-color: #F0FFF0; 
 padding: 1rem; 
 border-radius: 10px; 
 border-left: 5px solid #2E8B57; 
 margin: 1rem 0; 
 } 
 .compound-card { 
 background-color: #F8F9FA; 
 padding: 1rem; 
 border-radius: 8px; 
 margin: 0.5rem 0; 
 border: 1px solid #DEE2E6;
 } 
</style> 
""", unsafe_allow_html=True) 
# Title and description 
st.markdown('<h1 class="main-header">�� RemedicaAI</h1>', unsafe_allow_html=True) st.markdown("### Transform Waste into Wellness") 
st.markdown("Upload images of fruit/vegetable peels to discover their medicinal compounds and health benefits.") 
# Initialize session state 
if 'uploaded_image' not in st.session_state: 
 st.session_state.uploaded_image = None 
if 'results' not in st.session_state: 
 st.session_state.results =None 
if 'history' not in st.session_state: 
 st.session_state.history =[] 
# Sidebar 
with st.sidebar: 
 st.image("https://img.icons8.com/color/96/000000/recycle.png" , width=100) 
 st.markdown("### ��️ Configuration") 
  
 # API Keys (in production, use secrets management) 
 PUBMED_API_KEY = st.text_input("PubMed API Key", type="password", 
 value="YOUR_PUBMED_API_KEY_HERE") 
  
 st.markdown("---") 
 st.markdown("### �� Model Selection") 
 model_choice = st.selectbox( 
 "Select Model", 
 ["ResNet50", "EfficientNet", "Custom CNN"] 
 ) 
  
 st.markdown("---") 
 st.markdown("### �� Features") 
 st.markdown(""" 
 - �� Waste Recognition 
 - �� Medicinal Compounds 
 - �� PubMed Research 
 - �� Health Benefits 
 - �� History Tracking 
 """) 
  
 if st.button("Clear History"): 
 st.session_state.history =[] 
 st.success("History cleared!") 
# Main content area 
tab1, tab2, tab3, tab4 = st.tabs(["��Upload & Analyze", "�� Compounds Database", "�� Research", "�� History"]) 
with tab1: 
 col1, col2 = st.columns(2) 
  
 with col1:
 st.markdown("### �� Upload Waste Image") 
 uploaded_file = st.file_uploader( 
 "Choose an image of fruit/vegetable peel", 
 type=['jpg', 'jpeg', 'png'], 
 help="Supported: Banana, Apple, Orange, Potato, Onion, Garlic, etc."  ) 
  
 if uploaded_file is not None: 
 image = Image.open(uploaded_file) 
 st.image(image, caption="Uploaded Image", use_column_width=True)  st.session_state.uploaded_image = image 
  
 # Process button 
 if st.button("��Analyze Waste",type="primary"): 
 with st.spinner("Analyzing image and fetching data..."): 
 # Image classification would go here 
 # For demo, we'll simulate classification 
 waste_types ={ 
 'banana': 'Banana Peel', 
 'apple': 'Apple Peel', 
 'orange': 'Orange Peel', 
 'potato': 'Potato Peel', 
 'onion': 'Onion Peel', 
 'garlic': 'Garlic Peel' 
 } 
  
 # Simulate prediction (replace with actual model) 
 predicted_waste = "banana" # This would come from your model  waste_name = waste_types.get(predicted_waste, "Unknown Waste")   
 # Get compounds information 
 compounds_data =get_compounds_info(predicted_waste)   
 # Get PubMed research 
 pubmed_data = fetch_pubmed_research(waste_name,PUBMED_API_KEY)   
 # Store results 
 st.session_state.results ={ 
 'waste_name': waste_name, 
 'compounds': compounds_data, 
 'research': pubmed_data, 
 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  } 
  
 # Add to history 
 st.session_state.history.append(st.session_state.results)   
 st.success(f"✅Identified as: {waste_name}") 
  
 with col2: 
 if st.session_state.results: 
 st.markdown("### �� Analysis Results") 
  
 st.markdown(f"**Identified Waste:** {st.session_state.results['waste_name']}")  st.markdown(f"**Analysis Time:** {st.session_state.results['timestamp']}")
  
 st.markdown("---") 
 st.markdown("#### �� Key Medicinal Compounds")   
 compounds = st.session_state.results['compounds']  for compound in compounds: 
 with st.expander(f"�� {compound['name']}"): 
 st.markdown(f"**Type:** {compound['type']}")  st.markdown(f"**Health Benefits:** {compound['benefits']}")  st.markdown(f"**Applications:** {compound['applications']}")   
 st.markdown("---") 
 st.markdown("#### �� Research Highlights") 
  
 research = st.session_state.results['research'] 
 for i, article in enumerate(research[:3]): # Show top 3  st.markdown(f"**{i+1}. {article['title']}**") 
 st.markdown(f"*Journal:* {article['journal']}") 
 st.markdown(f"*Published:* {article['date']}") 
 st.markdown(f"[Read more]({article['link']})") 
# Compounds Database Tab 
with tab2: 
 st.markdown("### �� Medicinal Compounds Database")   
 # Database of common compounds 
 compounds_db = { 
 "Banana Peel": [ 
 {"name": "Dopamine", "type": "Neurotransmitter",  "benefits": "Mood regulation, antioxidant", 
 "applications": "Mental health, Parkinson's disease"},  {"name": "Serotonin", "type": "Neurotransmitter",  "benefits": "Sleep regulation, mood enhancement",  "applications": "Depression, anxiety disorders"}, 
 {"name": "Lutein", "type": "Carotenoid", 
 "benefits": "Eye health, antioxidant", 
 "applications": "Macular degeneration, cataracts"}  ], 
 "Apple Peel": [ 
 {"name": "Quercetin", "type": "Flavonoid", 
 "benefits": "Anti-inflammatory, antioxidant", 
 "applications": "Allergies, heart health"}, 
 {"name": "Ursolic Acid", "type": "Triterpenoid", 
 "benefits": "Anti-obesity, muscle growth", 
 "applications": "Weight management, fitness"} 
 ], 
 "Orange Peel": [ 
 {"name": "Limonene", "type": "Terpene", 
 "benefits": "Anti-cancer, antioxidant", 
 "applications": "Cancer prevention, aromatherapy"},  {"name": "Hesperidin", "type": "Flavonoid", 
 "benefits": "Cardiovascular protection", 
 "applications": "Blood pressure, circulation"} 
 ] 
 }
  
 # Display database 
 for waste, compounds in compounds_db.items(): 
 st.markdown(f"#### �� {waste}") 
 df = pd.DataFrame(compounds) 
 st.dataframe(df, use_container_width=True) 
# Research Tab 
with tab3: 
 st.markdown("### �� PubMed Research Interface") 
  
 search_term = st.text_input("Search PubMed for research:", "fruit peel medicinal compounds")   
 if st.button("Search PubMed"): 
 if PUBMED_API_KEY: 
 with st.spinner("Searching PubMed..."): 
 results = fetch_pubmed_research(search_term, PUBMED_API_KEY)   
 for article in results: 
 with st.container(): 
 st.markdown(f"**{article['title']}**") 
 st.markdown(f"*Authors:* {article['authors']}") 
 st.markdown(f"*Journal:* {article['journal']}") 
 st.markdown(f"*Abstract:* {article['abstract'][:300]}...") 
 st.markdown(f"[Full Article]({article['link']})") 
 st.markdown("---") 
 else: 
 st.warning("Please enter your PubMed API key in the sidebar" ) 
# History Tab 
with tab4: 
 st.markdown("### �� Analysis History") 
  
 if st.session_state.history: 
 for i, analysis in enumerate(reversed(st.session_state.history)): 
 with st.expander(f"Analysis {i+1}: {analysis['waste_name']} - {analysis['timestamp']}"):  st.markdown(f"**Waste:** {analysis['waste_name']}") 
 st.markdown(f"**Time:** {analysis['timestamp']}") 
  
 if analysis['compounds']: 
 st.markdown("**Compounds Found:**") 
 for compound in analysis['compounds']: 
 st.markdown(f"- {compound['name']}: {compound['benefits']}")  else: 
 st.info("No analysis history yet. Upload an image to get started!" ) 
# Functions 
def get_compounds_info(waste_type): 
 """Get medicinal compounds information for a waste type""" 
 compounds_db = { 
 'banana': [ 
 { 
 'name': 'Dopamine', 
 'type': 'Neurotransmitter', 
 'benefits': 'Mood regulation, antioxidant properties, cardiovascular health',
 'applications': 'Mental health disorders, Parkinson\'s disease'  }, 
 { 
 'name': 'Serotonin', 
 'type': 'Neurotransmitter', 
 'benefits': 'Sleep regulation, mood enhancement, appetite control',  'applications': 'Depression, anxiety, insomnia' 
 } 
 ], 
 'apple': [ 
 { 
 'name': 'Quercetin', 
 'type': 'Flavonoid', 
 'benefits': 'Anti-inflammatory, antioxidant, antiviral',  'applications': 'Allergies, heart disease, cancer prevention'  }, 
 { 
 'name': 'Ursolic Acid', 
 'type': 'Triterpenoid', 
 'benefits': 'Anti-obesity, anti-inflammatory, muscle growth',  'applications': 'Weight management, fitness, skincare'  } 
 ] 
 } 
  
 return compounds_db.get(waste_type,[]) 
def fetch_pubmed_research(search_term, api_key): 
 """Fetch research articles from PubMed""" 
 # Note: PubMed doesn't require an API key for basic searches  # This is a simplified version. For production, use Biopython or requests   
 base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"   
 params = { 
 'db': 'pubmed', 
 'term': f"{search_term} medicinal compounds", 
 'retmode': 'json', 
 'retmax': 5, 
 'sort': 'relevance' 
 } 
  
 try: 
 response = requests.get(base_url, params=params) 
  
 if response.status_code == 200: 
 data = response.json() 
  
 # For demo, return simulated data 
 # In production, you would fetch actual article details  return [ 
 { 
 'title': f"Medicinal properties of {search_term}",  'authors': "Smith J, et al.", 
 'journal': "Journal of Natural Products", 
 'date': "2023",
 'abstract': f"Study on the medicinal compounds found in {search_term}...",  'link': "https://pubmed.ncbi.nlm.nih.gov/example" 
 } 
 ] 
 else: 
 return [] 
 except: 
 # Return sample data if API fails 
 return [ 
 { 
 'title': "Phytochemical Analysis of Fruit Peels", 
 'authors': "Research Team", 
 'journal': "Journal of Agricultural Chemistry", 
 'date': "2023", 
 'abstract': "Analysis of medicinal compounds in various fruit peels..." ,  'link': "#" 
 } 
 ] 
# Footer 
st.markdown("---") 
st.markdown(""" 
<div style='text-align: center'> 
 <p>�� <b>RemedicaAI</b> - Transforming Food Waste into Medicinal Resources</p>  <p><small>Always consult healthcare professionals before using natural remedies</small></p> </div> 
""", unsafe_allow_html=True) 
2. Machine Learning Model (Image Classification) 
python 
Copy 
Download 
# model.py 
import torch 
import torch.nn as nn 
import torchvision.models as models 
import torchvision.transforms as transforms 
from PIL import Image 
import numpy as np 
class WasteClassifier: 
 def __init__(self, model_path=None): 
 self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
 self.model = self.load_model(model_path) 
 self.transform = self.get_transforms() 
 self.class_names = ['banana', 'apple', 'orange', 'potato', 'onion', 'garlic', 'other']   
 def load_model(self, model_path): 
 """Load pre-trained model or create new one""" 
 model = models.resnet50(pretrained=True) 
 num_features = model.fc.in_features 
 
 # Modify final layer for our classification 
 model.fc = nn.Sequential( 
 nn.Linear(num_features, 256), 
 nn.ReLU(), 
 nn.Dropout(0.3), 
 nn.Linear(256, 128), 
 nn.ReLU(), 
 nn.Dropout(0.2), 
 nn.Linear(128, len(self.class_names)) 
 ) 
  
 if model_path and os.path.exists(model_path): 
 model.load_state_dict(torch.load(model_path, map_location=self.device))   
 model = model.to(self.device) 
 model.eval() 
 return model 
  
 def get_transforms(self): 
 """Define image transformations""" 
 return transforms.Compose([ 
 transforms.Resize((224, 224)), 
 transforms.ToTensor(), 
 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
 std=[0.229, 0.224, 0.225]) 
 ]) 
  
 def predict(self, image_path): 
 """Predict waste type from image""" 
 image = Image.open(image_path).convert('RGB') 
 image_tensor = self.transform(image).unsqueeze(0).to(self.device)   
 with torch.no_grad(): 
 outputs =self.model(image_tensor) 
 probabilities = torch.nn.functional.softmax(outputs, dim=1)  confidence, predicted_idx = torch.max(probabilities, 1)   
 predicted_class = self.class_names[predicted_idx.item()]  confidence_score = confidence.item() 
  
 return predicted_class, confidence_score 
# Train script (simplified) 
def train_model(): 
 import torch.optim as optim 
 from torch.utils.data import DataLoader 
 from torchvision import datasets 
  
 # Dataset preparation (you'll need to collect your own images)  transform = transforms.Compose([ 
 transforms.Resize((224, 224)), 
 transforms.RandomHorizontalFlip(), 
 transforms.RandomRotation(10), 
 transforms.ToTensor(), 
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ])
  
 # Assuming you have dataset in 'data/train' and 'data/val' 
 train_dataset = datasets.ImageFolder('data/train', transform=transform)  val_dataset = datasets.ImageFolder('data/val', transform=transform) 
  
 train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) 
  
 # Training loop 
 classifier = WasteClassifier() 
 criterion = nn.CrossEntropyLoss() 
 optimizer = optim.Adam(classifier.model.parameters(), lr=0.001) 
  
 for epoch in range(10): 
 # Training code here 
 pass 
  
 # Save model 
 torch.save(classifier.model.state_dict(), 'waste_classifier.pth') 
3. PubMed API Integration (More Advanced) 
python 
Copy 
Download 
# pubmed_integration.py 
import requests 
from bs4 import BeautifulSoup 
import xml.etree.ElementTree as ET 
import pandas as pd 
from typing import List, Dict 
import time 
class PubMedAPI: 
 def __init__(self, email: str = "your_email@example.com"): 
 """ 
 Initialize PubMed API 
  
 Note: PubMed doesn't require an API key for most operations, 
 but they request you to provide an email for large queries 
 """ 
 self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils" 
 self.email = email 
  
 def search_articles(self, query: str, max_results: int = 10) -> List[Dict]:  """ 
 Search PubMed for articles 
  
 Args: 
 query: Search terms 
 max_results: Maximum number of results 
  
 Returns:
 List of article dictionaries 
 """ 
 # Search for article IDs 
 search_url = f"{self.base_url}/esearch.fcgi" 
 params = { 
 'db': 'pubmed', 
 'term': query, 
 'retmax': max_results, 
 'retmode': 'json', 
 'email': self.email 
 } 
  
 try: 
 response = requests.get(search_url, params=params)  data = response.json() 
  
 article_ids = data.get('esearchresult', {}).get('idlist', [])   
 if not article_ids: 
 return [] 
  
 # Fetch article details 
 return self.fetch_article_details(article_ids)   
 except Exception as e: 
 print(f"Error searching PubMed: {e}") 
 return [] 
  
 def fetch_article_details(self, article_ids: List[str]) -> List[Dict]:  """ 
 Fetch detailed information for articles 
  
 Args: 
 article_ids: List of PubMed IDs 
  
 Returns: 
 List of article details 
 """ 
 if not article_ids: 
 return [] 
  
 fetch_url = f"{self.base_url}/efetch.fcgi" 
 params = { 
 'db': 'pubmed', 
 'id': ','.join(article_ids), 
 'retmode': 'xml', 
 'email': self.email 
 } 
  
 try: 
 response = requests.get(fetch_url, params=params)  root = ET.fromstring(response.content) 
  
 articles = [] 
  
 for article in root.findall('.//PubmedArticle'):
 article_info = self.parse_article_xml(article) 
 if article_info: 
 articles.append(article_info) 
  
 return articles 
  
 except Exception as e: 
 print(f"Error fetching article details: {e}") 
 return [] 
  
 def parse_article_xml(self, article_xml) -> Dict: 
 """ 
 Parse XML article data 
  
 Args: 
 article_xml: XML element 
  
 Returns: 
 Dictionary with article information 
 """ 
 try: 
 # Extract title 
 title_elem = article_xml.find('.//ArticleTitle') 
 title = title_elem.text if title_elem is not None else "No title"   
 # Extract abstract 
 abstract_elem = article_xml.find('.//AbstractText') 
 abstract = abstract_elem.text if abstract_elem is not None else "No abstract"   
 # Extract authors 
 authors = [] 
 for author in article_xml.findall('.//Author'): 
 last_name = author.find('LastName') 
 fore_name = author.find('ForeName') 
 if last_name is not None and fore_name is not None: 
 authors.append(f"{fore_name.text} {last_name.text}")   
 # Extract journal 
 journal_elem = article_xml.find('.//Journal/Title') 
 journal = journal_elem.text if journal_elem is not None else "Unknown journal"   
 # Extract publication date 
 pub_date_elem = article_xml.find('.//PubDate/Year') 
 pub_date = pub_date_elem.text if pub_date_elem is not None else "Unknown"   
 # Extract PubMed ID 
 pmid_elem = article_xml.find('.//PMID') 
 pmid = pmid_elem.text if pmid_elem is not None else "" 
  
 return { 
 'title': title, 
 'abstract': abstract, 
 'authors': ', '.join(authors) if authors else "Unknown authors",  'journal': journal, 
 'publication_date': pub_date, 
 'pmid': pmid,
 'link': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" 
 } 
  
 except Exception as e: 
 print(f"Error parsing article: {e}") 
 return None 
  
 def get_compound_research(self, compound_name:str) -> List[Dict]: 
 """ 
 Get research articles for specific compound 
  
 Args: 
 compound_name: Name of medicinal compound 
  
 Returns: 
 List of research articles 
 """ 
 query = f"{compound_name}[Title/Abstract] AND medicinal[Title/Abstract]"  return self.search_articles(query) 
  
 def get_waste_research(self,waste_name:str) -> List[Dict]: 
 """ 
 Get research articles for specific waste material 
  
 Args: 
 waste_name: Name of fruit/vegetable waste 
  
 Returns: 
 List of research articles 
 """ 
 query = f'"{waste_name} peel"[Title/Abstract] AND (phytochemical OR medicinal OR compound)'  return self.search_articles(query) 
# Example usage 
if __name__ == "__main__": 
 pubmed = PubMedAPI(email="your_email@example.com") 
  
 # Search for banana peel research 
 articles = pubmed.get_waste_research("banana") 
  
 for article in articles[:3]: 
 print(f"Title: {article['title']}") 
 print(f"Authors: {article['authors']}") 
 print(f"Link: {article['link']}") 
 print("-" * 50) 
4. Requirements File 
txt 
Copy 
Download 
# requirements.txt 
streamlit==1.28.0
torch==2.0.1 
torchvision==0.15.2 
pillow==10.0.0 
requests==2.31.0 
pandas==2.0.3 
numpy==1.24.3 
matplotlib==3.7.2 
beautifulsoup4==4.12.2 
lxml==4.9.3 
scikit-learn==1.3.0 
plotly==5.17.0 
5. Installation and Setup Instructions 
bash 
Copy 
Download 
# Setup instructions 
# 1. Create virtual environment 
python -m venv remedicaai_env 
source remedicaai_env/bin/activate # On Windows: remedicaai_env\Scripts\activate 
# 2. Install dependencies 
pip install -r requirements.txt 
# 3. Create directory structure 
mkdir -p data/train data/val data/test models 
# 4. Download sample dataset or create your own 
# You'll need to collect images of different fruit/vegetable peels 
# 5. Train the model (optional) 
python train_model.py 
# 6. Run the application 
streamlit run app.py 
6. Alternative: Using Google's Medicinal Plant Dataset  API 
python 
Copy 
Download 
# alternative_api.py 
import requests 
def get_medicinal_info_plantnet(waste_name, api_key): 
 """
 Alternative: Use PlantNet API for plant identification 
 """ 
 url = "https://my-api.plantnet.org/v2/identify/all" 
  
 # This would require actual image upload 
 params = { 
 'api-key': api_key 
 } 
  
 return { 
 'name': waste_name, 
 'family': 'Rosaceae', # Example 
 'medicinal_uses': 'Rich in antioxidants and anti-inflammatory compounds',  'compounds': ['Quercetin', 'Flavonoids', 'Vitamins'] 
 }
