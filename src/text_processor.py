"""
Text Extraction and Chunking System
Extracts text from PDFs and chunks into 200-500 word segments
"""
import os
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Try importing PDF processing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False


@dataclass
class TextChunk:
    """Text chunk data structure"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    word_count: int
    source_file: str
    page_number: Optional[int] = None


class TextProcessor:
    """Process PDFs and extract/chunk text"""
    
    def __init__(self, 
                 pdf_directory: str = None,
                 chunk_size_range: tuple = (200, 500),
                 overlap_words: int = 50):
        
        self.pdf_directory = pdf_directory or "/home/ah0012/project/data/pdfs"
        self.min_chunk_size = chunk_size_range[0]
        self.max_chunk_size = chunk_size_range[1]
        self.overlap_words = overlap_words
        
        # Check available libraries
        if not PYMUPDF_AVAILABLE and not PDFMINER_AVAILABLE:
            print("Warning: Neither PyMuPDF nor pdfminer.six available. Using mock extraction.")
    
    def extract_text_from_pdf(self, pdf_path: str, method: str = "auto") -> Dict[str, Any]:
        """
        Extract text from PDF using available libraries
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('pymupdf', 'pdfminer', 'auto')
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        filename = os.path.basename(pdf_path)
        
        # Determine extraction method
        if method == "auto":
            if PYMUPDF_AVAILABLE:
                method = "pymupdf"
            elif PDFMINER_AVAILABLE:
                method = "pdfminer"
            else:
                method = "mock"
        
        try:
            if method == "pymupdf" and PYMUPDF_AVAILABLE:
                return self._extract_with_pymupdf(pdf_path)
            elif method == "pdfminer" and PDFMINER_AVAILABLE:
                return self._extract_with_pdfminer(pdf_path)
            else:
                return self._extract_mock(pdf_path)
        except Exception as e:
            print(f"Error extracting text from {filename}: {e}")
            return self._extract_mock(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using PyMuPDF"""
        doc = fitz.open(pdf_path)
        pages = []
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            pages.append({
                "page_number": page_num + 1,
                "text": page_text,
                "word_count": len(page_text.split())
            })
            full_text += page_text + "\n\n"
        
        doc.close()
        
        return {
            "filename": os.path.basename(pdf_path),
            "full_text": full_text.strip(),
            "pages": pages,
            "total_pages": len(pages),
            "total_words": len(full_text.split()),
            "extraction_method": "pymupdf"
        }
    
    def _extract_with_pdfminer(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using pdfminer.six"""
        # Configure layout analysis
        laparams = LAParams(
            line_margin=0.5,
            char_margin=2.0,
            word_margin=0.1,
            boxes_flow=0.5
        )
        
        text = extract_text(pdf_path, laparams=laparams)
        
        return {
            "filename": os.path.basename(pdf_path),
            "full_text": text.strip(),
            "pages": [{"page_number": 1, "text": text, "word_count": len(text.split())}],
            "total_pages": 1,  # pdfminer doesn't easily separate pages
            "total_words": len(text.split()),
            "extraction_method": "pdfminer"
        }
    
    def _extract_mock(self, pdf_path: str) -> Dict[str, Any]:
        """Generate mock extracted text for development"""
        filename = os.path.basename(pdf_path)
        
        # Generate mock content based on filename
        mock_content = self._generate_mock_content(filename)
        
        return {
            "filename": filename,
            "full_text": mock_content,
            "pages": [{"page_number": 1, "text": mock_content, "word_count": len(mock_content.split())}],
            "total_pages": 1,
            "total_words": len(mock_content.split()),
            "extraction_method": "mock"
        }
    
    def _generate_mock_content(self, filename: str) -> str:
        """Generate mock content based on filename"""
        mock_contents = {
            "return_policy.pdf": """
            Return and Refund Policy
            
            We want you to be completely satisfied with your purchase. If you're not happy with your order, 
            you can return most items within 30 days of delivery for a full refund.
            
            Return Eligibility:
            Items must be in their original condition with all tags attached. Items that have been worn, 
            damaged, or altered cannot be returned. Certain items like personalized products, intimate 
            apparel, and perishable goods are not eligible for return.
            
            How to Return:
            1. Log into your account and select the order you want to return
            2. Choose the items you want to return and select a reason
            3. Print your prepaid return label
            4. Package your items securely in the original packaging if possible
            5. Drop off your package at any authorized shipping location
            
            Refund Processing:
            Once we receive your return, we'll inspect the items and process your refund within 5-7 business days. 
            Refunds will be credited to your original payment method. Please note that original shipping costs 
            are non-refundable unless the return is due to our error.
            """,
            
            "shipping_guide.pdf": """
            Shipping Information Guide
            
            We offer several shipping options to meet your needs. All orders are processed within 1-2 business days 
            and ship Monday through Friday, excluding holidays.
            
            Shipping Options:
            Standard Shipping (5-7 business days) - Free on orders over $50
            Expedited Shipping (2-3 business days) - $9.99
            Express Shipping (1-2 business days) - $19.99
            Overnight Shipping (next business day) - $29.99
            
            Order Tracking:
            Once your order ships, you'll receive a confirmation email with tracking information. You can track 
            your package using the tracking number provided or by logging into your account.
            
            International Shipping:
            We currently ship to over 100 countries worldwide. International shipping costs and delivery times 
            vary by destination. Customers are responsible for any customs duties or taxes that may apply.
            
            Delivery Information:
            Packages are typically delivered between 9 AM and 8 PM, Monday through Saturday. A signature may 
            be required for high-value orders. If you're not available to receive your package, the carrier 
            will leave a delivery notice with instructions for redelivery or pickup.
            """,
            
            "customer_service_faq.pdf": """
            Frequently Asked Questions
            
            Order Questions:
            Q: How can I track my order?
            A: You can track your order using the tracking number sent to your email, or by logging into your account.
            
            Q: Can I change or cancel my order?
            A: Orders can be modified or cancelled within 1 hour of placement. After that, please contact customer service.
            
            Q: What payment methods do you accept?
            A: We accept all major credit cards, PayPal, Apple Pay, and gift cards.
            
            Account Questions:
            Q: How do I reset my password?
            A: Click "Forgot Password" on the login page and follow the instructions sent to your email.
            
            Q: How do I update my account information?
            A: Log into your account and go to "Account Settings" to update your information.
            
            Product Questions:
            Q: Are your products authentic?
            A: Yes, we only sell authentic products from authorized retailers and manufacturers.
            
            Q: Do you offer price matching?
            A: We offer price matching on identical items from qualifying competitors. Contact customer service for details.
            
            Technical Issues:
            Q: The website isn't working properly. What should I do?
            A: Try clearing your browser cache and cookies, or try a different browser. If problems persist, contact our technical support team.
            """
        }
        
        # Return specific content or generic content
        return mock_contents.get(filename, f"""
        This document contains important information about our e-commerce services and policies.
        
        We are committed to providing excellent customer service and transparent business practices.
        Our team works hard to ensure that every customer has a positive shopping experience.
        
        For more detailed information, please visit our website or contact our customer service team.
        We are available Monday through Friday from 9 AM to 6 PM EST to assist with any questions or concerns.
        
        Thank you for choosing our services. We appreciate your business and look forward to serving you.
        """)
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        Chunk text into segments of 200-500 words with overlap
        
        Args:
            text: Full text to chunk
            metadata: Metadata about the source document
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Create chunks
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_number = 1
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # Check if adding this sentence would exceed max chunk size
            if current_word_count + sentence_words > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                if current_word_count >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_text, chunk_number, metadata, current_word_count
                    ))
                    chunk_number += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_word_count = len(" ".join(current_chunk).split())
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        # Add final chunk if it meets minimum size
        if current_chunk and current_word_count >= self.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text, chunk_number, metadata, current_word_count
            ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting on periods, exclamation marks, and question marks
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(self, current_chunk: List[str]) -> List[str]:
        """Get overlap sentences for next chunk"""
        if not current_chunk:
            return []
        
        overlap_text = " ".join(current_chunk)
        overlap_words = overlap_text.split()
        
        if len(overlap_words) <= self.overlap_words:
            return current_chunk
        
        # Take last overlap_words worth of content
        last_words = overlap_words[-self.overlap_words:]
        overlap_text = " ".join(last_words)
        
        # Try to end at a sentence boundary
        sentences = self._split_into_sentences(overlap_text)
        return sentences
    
    def _create_chunk(self, text: str, chunk_number: int, metadata: Dict[str, Any], word_count: int) -> TextChunk:
        """Create a TextChunk object"""
        chunk_id = f"{metadata.get('filename', 'unknown')}_{chunk_number:03d}"
        
        chunk_metadata = {
            **metadata,
            "chunk_number": chunk_number,
            "created_at": datetime.now().isoformat()
        }
        
        return TextChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=chunk_metadata,
            word_count=word_count,
            source_file=metadata.get('filename', 'unknown')
        )
    
    def process_all_pdfs(self) -> List[TextChunk]:
        """Process all PDFs in the directory and return chunks"""
        all_chunks = []
        
        # Get list of PDF files
        pdf_files = []
        if os.path.exists(self.pdf_directory):
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found. Generating mock chunks for testing.")
            return self._generate_mock_chunks()
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            
            try:
                # Extract text
                extraction_result = self.extract_text_from_pdf(pdf_path)
                
                # Determine topic from filename
                topic = self._determine_topic(pdf_file)
                
                # Create metadata
                metadata = {
                    "filename": pdf_file,
                    "topic": topic,
                    "source": "pdf_extraction",
                    "total_pages": extraction_result["total_pages"],
                    "total_words": extraction_result["total_words"],
                    "extraction_method": extraction_result["extraction_method"]
                }
                
                # Chunk text
                chunks = self.chunk_text(extraction_result["full_text"], metadata)
                all_chunks.extend(chunks)
                
                print(f"Processed {pdf_file}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _determine_topic(self, filename: str) -> str:
        """Determine topic from filename"""
        topic_mapping = {
            "return": "returns",
            "shipping": "shipping", 
            "faq": "support",
            "customer": "support",
            "privacy": "legal",
            "terms": "legal",
            "product": "products",
            "care": "products",
            "size": "products",
            "payment": "payments",
            "security": "security",
            "loyalty": "rewards",
            "gift": "payments",
            "warranty": "warranties",
            "international": "shipping"
        }
        
        filename_lower = filename.lower()
        for keyword, topic in topic_mapping.items():
            if keyword in filename_lower:
                return topic
        
        return "general"
    
    def _generate_mock_chunks(self) -> List[TextChunk]:
        """Generate mock chunks for testing"""
        mock_chunks = []
        
        mock_documents = [
            {
                "filename": "return_policy.pdf",
                "topic": "returns",
                "text": "Our return policy allows returns within 30 days of purchase. Items must be in original condition with tags attached. Refunds are processed within 5-7 business days. Certain items like personalized products and intimate apparel cannot be returned."
            },
            {
                "filename": "shipping_guide.pdf", 
                "topic": "shipping",
                "text": "We offer multiple shipping options including standard, expedited, and express delivery. Free shipping is available on orders over $50. International shipping is available to over 100 countries with varying delivery times and costs."
            },
            {
                "filename": "customer_faq.pdf",
                "topic": "support", 
                "text": "Frequently asked questions cover order tracking, payment methods, account management, and technical support. Our customer service team is available Monday through Friday to assist with any questions or concerns you may have."
            }
        ]
        
        for i, doc in enumerate(mock_documents, 1):
            metadata = {
                "filename": doc["filename"],
                "topic": doc["topic"],
                "source": "mock_data",
                "created_at": datetime.now().isoformat()
            }
            
            chunk = TextChunk(
                chunk_id=f"{doc['filename']}_{i:03d}",
                text=doc["text"],
                metadata=metadata,
                word_count=len(doc["text"].split()),
                source_file=doc["filename"]
            )
            mock_chunks.append(chunk)
        
        return mock_chunks


def main():
    """Main function to process PDFs and create chunks"""
    processor = TextProcessor()
    chunks = processor.process_all_pdfs()
    
    print(f"\nCreated {len(chunks)} text chunks:")
    for chunk in chunks[:5]:  # Show first 5 chunks
        print(f"- {chunk.chunk_id}: {chunk.word_count} words from {chunk.source_file}")
    
    if len(chunks) > 5:
        print(f"... and {len(chunks) - 5} more chunks")
    
    return chunks


if __name__ == "__main__":
    main()