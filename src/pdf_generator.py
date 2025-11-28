"""
PDF Generation Pipeline for creating e-commerce documents
Generates 6-15 PDFs using LLM-generated text and Python tools
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_factory import get_default_processor


class PDFGenerator:
    """Generate e-commerce related PDFs with LLM-generated content"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or "/home/ah0012/project/data/pdfs"
        self.llm_processor = get_default_processor()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # PDF topics and templates
        self.pdf_topics = [
            {
                "filename": "return_policy",
                "title": "Return and Refund Policy",
                "topic": "returns",
                "prompt": "Write a comprehensive return and refund policy for an e-commerce company. Include time limits, conditions, excluded items, and refund processing times."
            },
            {
                "filename": "shipping_guide",
                "title": "Shipping Information Guide", 
                "topic": "shipping",
                "prompt": "Create a detailed shipping guide covering shipping methods, delivery times, costs, international shipping, and tracking information."
            },
            {
                "filename": "customer_service_faq",
                "title": "Customer Service FAQ",
                "topic": "support",
                "prompt": "Generate a comprehensive FAQ covering common customer service questions about orders, payments, accounts, and technical issues."
            },
            {
                "filename": "privacy_policy",
                "title": "Privacy Policy",
                "topic": "legal",
                "prompt": "Write a privacy policy explaining how customer data is collected, used, stored, and protected in an e-commerce environment."
            },
            {
                "filename": "terms_of_service",
                "title": "Terms of Service",
                "topic": "legal", 
                "prompt": "Create terms of service for an e-commerce platform covering user responsibilities, prohibited uses, and service limitations."
            },
            {
                "filename": "product_care_guide",
                "title": "Product Care and Maintenance Guide",
                "topic": "products",
                "prompt": "Write a guide on how to care for and maintain different types of products sold online, including electronics, clothing, and home goods."
            },
            {
                "filename": "size_guide",
                "title": "Size and Fit Guide",
                "topic": "products",
                "prompt": "Create a comprehensive sizing guide for clothing, shoes, and accessories with measurement instructions and size charts."
            },
            {
                "filename": "payment_security",
                "title": "Payment Security Information",
                "topic": "security",
                "prompt": "Explain payment security measures, accepted payment methods, and how customer payment information is protected."
            },
            {
                "filename": "loyalty_program",
                "title": "Loyalty Program Benefits",
                "topic": "rewards",
                "prompt": "Describe a customer loyalty program including how to earn points, redeem rewards, and member benefits."
            },
            {
                "filename": "gift_card_policy",
                "title": "Gift Card Terms and Conditions",
                "topic": "payments",
                "prompt": "Create terms and conditions for gift cards including purchase limits, expiration, and usage restrictions."
            },
            {
                "filename": "warranty_information",
                "title": "Product Warranty Information", 
                "topic": "warranties",
                "prompt": "Write warranty information covering different product categories, warranty periods, and claim procedures."
            },
            {
                "filename": "international_shipping",
                "title": "International Shipping Guidelines",
                "topic": "shipping",
                "prompt": "Create guidelines for international shipping including customs, duties, restrictions, and delivery timeframes."
            }
        ]
    
    def generate_all_pdfs(self) -> List[Dict[str, Any]]:
        """Generate all PDF documents"""
        generated_pdfs = []
        
        # Generate 6-15 PDFs (let's do 10)
        selected_topics = self.pdf_topics[:10]
        
        for topic_info in selected_topics:
            try:
                pdf_path = self.generate_pdf(topic_info)
                generated_pdfs.append({
                    "filename": f"{topic_info['filename']}.pdf",
                    "path": pdf_path,
                    "title": topic_info["title"],
                    "topic": topic_info["topic"],
                    "created_at": datetime.now().isoformat()
                })
                print(f"Generated: {topic_info['filename']}.pdf")
            except Exception as e:
                print(f"Error generating {topic_info['filename']}: {e}")
        
        return generated_pdfs
    
    def generate_pdf(self, topic_info: Dict[str, Any]) -> str:
        """Generate a single PDF document"""
        # Generate content using LLM
        content = self.generate_content(topic_info["prompt"])
        
        # Create PDF
        pdf_path = os.path.join(self.output_dir, f"{topic_info['filename']}.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph(topic_info["title"], title_style))
        story.append(Spacer(1, 20))
        
        # Content
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), styles['Normal']))
                story.append(Spacer(1, 12))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d')}"
        story.append(Paragraph(footer_text, styles['Italic']))
        
        # Build PDF
        doc.build(story)
        
        return pdf_path
    
    def generate_content(self, prompt: str) -> str:
        """Generate content using LLM"""
        messages = [
            {
                "role": "system",
                "content": "You are a professional content writer specializing in e-commerce documentation. Write clear, comprehensive, and customer-friendly content."
            },
            {
                "role": "user",
                "content": prompt + "\n\nPlease provide well-structured content with clear sections and bullet points where appropriate."
            }
        ]
        
        try:
            content = self.llm_processor.generate_completion(messages, max_tokens=1500)
            return content
        except Exception as e:
            print(f"Error generating content: {e}")
            # Fallback content
            return f"""
            This document covers important information about our e-commerce services.
            
            Key Points:
            • We are committed to providing excellent customer service
            • Our policies are designed to protect both customers and our business
            • Please contact customer support if you have any questions
            • All policies are subject to change with notice
            
            For more information, please visit our website or contact our customer service team.
            
            Contact Information:
            Email: support@ecommerce.com
            Phone: 1-800-SHOP-NOW
            Hours: Monday-Friday 9AM-6PM EST
            """
    
    def get_generated_files(self) -> List[str]:
        """Get list of generated PDF files"""
        pdf_files = []
        for file in os.listdir(self.output_dir):
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(self.output_dir, file))
        return pdf_files


def main():
    """Main function to generate PDFs"""
    generator = PDFGenerator()
    generated_pdfs = generator.generate_all_pdfs()
    
    print(f"\nGenerated {len(generated_pdfs)} PDF files:")
    for pdf_info in generated_pdfs:
        print(f"- {pdf_info['title']} ({pdf_info['filename']})")
    
    return generated_pdfs


if __name__ == "__main__":
    main()