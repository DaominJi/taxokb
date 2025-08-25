import requests
import time
import json
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaperInformationCollector:
    """
    Collect paper information from Semantic Scholar API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the collector with optional API key for higher rate limits
        
        Args:
            api_key: Semantic Scholar API key (optional)
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {}
        
        if api_key:
            self.headers['x-api-key'] = api_key
            logger.info("Using Semantic Scholar API with authentication")
        else:
            logger.info("Using Semantic Scholar API without authentication (rate limited)")
        
        # Rate limiting (100 requests per 5 minutes for unauthenticated)
        self.request_delay = 5.0  # seconds between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    def search_paper_by_title(self, title: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Search for a paper by title and return the best match
        
        Args:
            title: Paper title to search for
            
        Returns:
            Paper data dictionary or None if not found
        """
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                # Clean the title for search
                clean_title = re.sub(r'[^\w\s]', ' ', title)
                clean_title = ' '.join(clean_title.split())
                
                # Search endpoint
                search_url = f"{self.base_url}/paper/search"
                
                params = {
                    'query': clean_title,
                    'limit': 5,
                    'fields': 'paperId,title,year,authors,venue,citationCount,referenceCount,publicationDate,abstract'
                }
                
                response = requests.get(search_url, params=params, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    papers = data.get('data', [])
                    
                    if papers:
                        # Find best match by comparing titles
                        best_match = self._find_best_title_match(title, papers)
                        if best_match:
                            logger.info(f"Found paper: {best_match.get('title', 'Unknown')}")
                            return best_match
                        else:
                            logger.warning(f"No good match found for title: {title}")
                    else:
                        logger.warning(f"No papers found for title: {title}")
                    
                    # If we got a successful response but no results, don't retry
                    return None
                    
                elif response.status_code == 429:
                    # Rate limit exceeded, wait longer before retry
                    wait_time = (attempt + 1) * 10  # Exponential backoff
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    
                elif response.status_code >= 500:
                    # Server error, retry with backoff
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Server error {response.status_code}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    
                else:
                    # Client error (4xx except 429), don't retry
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.ConnectionError as e:
                wait_time = (attempt + 1) * 5
                logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}. Waiting {wait_time} seconds...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    
            except Exception as e:
                logger.error(f"Error searching for paper '{title}': {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                    time.sleep(5)
                else:
                    logger.error(f"Failed after {max_retries} attempts")
        
        return None
    
    def _find_best_title_match(self, query_title: str, papers: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching paper from search results
        
        Args:
            query_title: Original title being searched
            papers: List of paper results from API
            
        Returns:
            Best matching paper or None
        """
        query_lower = query_title.lower().strip()
        
        # Remove common punctuation for comparison
        query_normalized = re.sub(r'[^\w\s]', '', query_lower)
        
        best_match = None
        best_score = 0
        
        for paper in papers:
            paper_title = paper.get('title', '').lower().strip()
            paper_normalized = re.sub(r'[^\w\s]', '', paper_title)
            
            # Calculate similarity score
            score = self._calculate_title_similarity(query_normalized, paper_normalized)
            
            if score > best_score and score > 0.8:  # Threshold for matching
                best_score = score
                best_match = paper
        
        return best_match
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two titles
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0
        
        # Also consider if one title contains the other
        contains_score = 0
        if title1 in title2 or title2 in title1:
            contains_score = 0.9
        
        return max(jaccard, contains_score)
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed paper information by Semantic Scholar paper ID
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Paper data dictionary or None if not found
        """
        try:
            self._rate_limit()
            
            # Paper details endpoint
            paper_url = f"{self.base_url}/paper/{paper_id}"
            
            params = {
                'fields': 'paperId,title,year,authors,venue,citationCount,referenceCount,publicationDate,abstract,citations,references'
            }
            
            response = requests.get(paper_url, params=params, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get paper {paper_id}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting paper {paper_id}: {e}")
        
        return None
    
    def extract_paper_info(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant information from paper data
        
        Args:
            paper_data: Raw paper data from API
            
        Returns:
            Formatted paper information
        """
        if not paper_data:
            return {}
        
        # Extract authors
        authors = []
        for author in paper_data.get('authors', []):
            author_name = author.get('name', '')
            if author_name:
                authors.append(author_name)
        
        # Format the extracted information
        info = {
            'semantic_scholar_id': paper_data.get('paperId', ''),
            'title': paper_data.get('title', ''),
            'year': paper_data.get('year'),
            'publication_date': paper_data.get('publicationDate', ''),
            'authors': authors,
            'authors_string': ', '.join(authors),
            'venue': paper_data.get('venue', ''),
            'citation_count': paper_data.get('citationCount', 0),
            'reference_count': paper_data.get('referenceCount', 0),
            'abstract': paper_data.get('abstract', '')
        }
        
        return info
    
    def collect_papers_info(self, titles: List[str]) -> List[Dict[str, Any]]:
        """
        Collect information for multiple papers
        
        Args:
            titles: List of paper titles
            
        Returns:
            List of paper information dictionaries
        """
        papers_info = []
        
        for i, title in enumerate(titles, 1):
            logger.info(f"Processing paper {i}/{len(titles)}: {title}")
            
            # Search for paper by title
            paper_data = self.search_paper_by_title(title.replace(':', '-'))
            
            if paper_data:
                info = self.extract_paper_info(paper_data)
                info['original_title'] = title  # Keep original title for reference
                papers_info.append(info)
            else:
                logger.warning(f"Could not find information for: {title}")
                papers_info.append({
                    'original_title': title,
                    'title': title,
                    'error': 'Paper not found in Semantic Scholar'
                })
        
        return papers_info
    
    def update_paper_register(self, register_path: str = 'data/paper_register.csv'):
        """
        Update the paper register CSV with collected information
        
        Args:
            register_path: Path to the paper register CSV file
        """
        try:
            # Load existing register
            df = pd.read_csv(register_path)
            
            # Get list of titles
            titles = df['title'].tolist()
            
            logger.info(f"Collecting information for {len(titles)} papers...")
            
            # Collect paper information
            papers_info = self.collect_papers_info(titles)
            
            # Create a mapping from title to info
            info_map = {info['original_title']: info for info in papers_info}
            
            # Add new columns to dataframe
            if 'year' not in df.columns:
                df['year'] = None
            if 'authors' not in df.columns:
                df['authors'] = None
            if 'venue' not in df.columns:
                df['venue'] = None
            if 'citation_count' not in df.columns:
                df['citation_count'] = None
            if 'semantic_scholar_id' not in df.columns:
                df['semantic_scholar_id'] = None
            
            # Update dataframe with collected information
            for idx, row in df.iterrows():
                title = row['title']
                if title in info_map:
                    info = info_map[title]
                    df.at[idx, 'year'] = info.get('year')
                    df.at[idx, 'authors'] = info.get('authors_string')
                    df.at[idx, 'venue'] = info.get('venue')
                    df.at[idx, 'citation_count'] = info.get('citation_count')
                    df.at[idx, 'semantic_scholar_id'] = info.get('semantic_scholar_id')
            
            # Save updated register
            df.to_csv(register_path, index=False)
            logger.info(f"Updated paper register saved to {register_path}")
            
            # Also save detailed information as JSON
            json_path = register_path.replace('.csv', '_detailed.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(papers_info, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed information saved to {json_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error updating paper register: {e}")
            return None
    
    def save_to_json(self, papers_info: List[Dict[str, Any]], output_path: str = 'output/papers_info.json'):
        """
        Save collected paper information to JSON file
        
        Args:
            papers_info: List of paper information dictionaries
            output_path: Path to save the JSON file
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(papers_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Paper information saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")


def main():
    """
    Main function to demonstrate usage
    """
    # Initialize collector (add API key if you have one)
    # To get an API key: https://www.semanticscholar.org/product/api
    # collector = PaperInformationCollector(api_key="YOUR_API_KEY_HERE")
    collector = PaperInformationCollector()
    
    # Option 1: Update existing paper register
    logger.info("Updating paper register with Semantic Scholar information...")
    updated_df = collector.update_paper_register('data/paper_register.csv')
    
    if updated_df is not None:
        print("\nUpdated Paper Register:")
        print(updated_df[['title', 'year', 'venue', 'citation_count']].head())
    
    # Option 2: Collect information for specific papers
    
    paper_register = pd.read_csv('data/paper_register.csv')
    titles = paper_register['title'].tolist()
    sample_titles = titles
    
    logger.info("\nCollecting information for sample papers...")
    papers_info = collector.collect_papers_info(sample_titles)
    
    # Save to JSON
    collector.save_to_json(papers_info, 'output/sample_papers_info.json')
    
    # Display results
    for paper in papers_info:
        print(f"\nTitle: {paper.get('title', 'Unknown')}")
        print(f"Year: {paper.get('year', 'Unknown')}")
        print(f"Authors: {paper.get('authors_string', 'Unknown')}")
        print(f"Venue: {paper.get('venue', 'Unknown')}")
        print(f"Citations: {paper.get('citation_count', 0)}")


if __name__ == "__main__":
    main()