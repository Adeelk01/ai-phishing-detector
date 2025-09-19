import gradio as gr
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import warnings
warnings.filterwarnings('ignore')

class PhishingDetector:
    def __init__(self):
        # Initialize the sentiment pipeline (we'll use this as base and adapt)
        self.device = 0 if torch.cuda.is_available() else -1
        
        # For demo purposes, we'll use a sentiment model and adapt it
        # In production, you'd fine-tune specifically on phishing data
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.classifier = pipeline("sentiment-analysis", 
                                 model=self.model_name, 
                                 device=self.device)
        
        # Phishing keywords and patterns
        self.phishing_keywords = [
            'urgent', 'verify', 'suspend', 'limited time', 'act now',
            'click here', 'confirm identity', 'update payment',
            'security alert', 'account locked', 'expired',
            'winner', 'congratulations', 'prize', 'lottery',
            'free money', 'inheritance', 'tax refund',
            'bitcoin', 'cryptocurrency', 'investment opportunity'
        ]
        
        self.suspicious_domains = [
            'bit.ly', 'tinyurl.com', 'shorturl.at', 't.co'
        ]
        
    def extract_features(self, email_text, subject="", sender=""):
        """Extract various features from email content"""
        features = {}
        
        # Text-based features
        features['length'] = len(email_text)
        features['word_count'] = len(email_text.split())
        features['exclamation_count'] = email_text.count('!')
        features['question_count'] = email_text.count('?')
        features['capital_ratio'] = sum(1 for c in email_text if c.isupper()) / max(len(email_text), 1)
        
        # URL analysis
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_text)
        features['url_count'] = len(urls)
        features['has_suspicious_domain'] = any(domain in email_text.lower() for domain in self.suspicious_domains)
        
        # Phishing keyword analysis
        text_lower = email_text.lower() + " " + subject.lower()
        features['phishing_keywords_count'] = sum(1 for keyword in self.phishing_keywords if keyword in text_lower)
        
        # Urgency indicators
        urgency_words = ['urgent', 'immediate', 'asap', 'expire', 'deadline', 'limited time']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text_lower)
        
        return features
    
    def analyze_email(self, email_text, subject="", sender=""):
        """Main analysis function"""
        if not email_text.strip():
            return "Please enter email content to analyze.", 0.0, ""
        
        # Extract features
        features = self.extract_features(email_text, subject, sender)
        
        # Get AI sentiment (as proxy for suspiciousness)
        try:
            result = self.classifier(email_text[:512])  # Limit text length
            sentiment_score = result[0]['score'] if result[0]['label'] == 'NEGATIVE' else 1 - result[0]['score']
        except:
            sentiment_score = 0.5
        
        # Calculate phishing probability
        risk_score = 0.0
        
        # Feature-based scoring
        if features['phishing_keywords_count'] > 0:
            risk_score += min(features['phishing_keywords_count'] * 0.15, 0.4)
        
        if features['urgency_score'] > 0:
            risk_score += min(features['urgency_score'] * 0.1, 0.2)
        
        if features['has_suspicious_domain']:
            risk_score += 0.2
        
        if features['url_count'] > 3:
            risk_score += 0.15
        
        if features['capital_ratio'] > 0.3:
            risk_score += 0.1
        
        if features['exclamation_count'] > 2:
            risk_score += 0.05
        
        # Combine with AI sentiment
        risk_score = (risk_score + sentiment_score) / 2
        risk_score = min(risk_score, 1.0)
        
        # Generate explanation
        explanation = self.generate_explanation(features, risk_score)
        
        # Risk level
        if risk_score >= 0.7:
            risk_level = "🔴 HIGH RISK - Likely Phishing"
        elif risk_score >= 0.4:
            risk_level = "🟡 MEDIUM RISK - Suspicious"
        else:
            risk_level = "🟢 LOW RISK - Appears Safe"
        
        return risk_level, round(risk_score * 100, 1), explanation
    
    def generate_explanation(self, features, risk_score):
        """Generate explanation for the risk assessment"""
        explanations = []
        
        if features['phishing_keywords_count'] > 0:
            explanations.append(f"• Contains {features['phishing_keywords_count']} phishing-related keywords")
        
        if features['urgency_score'] > 0:
            explanations.append(f"• Uses {features['urgency_score']} urgency-inducing words")
        
        if features['has_suspicious_domain']:
            explanations.append("• Contains suspicious shortened URLs")
        
        if features['url_count'] > 3:
            explanations.append(f"• Contains {features['url_count']} URLs (high count)")
        
        if features['capital_ratio'] > 0.3:
            explanations.append("• Excessive use of capital letters")
        
        if features['exclamation_count'] > 2:
            explanations.append(f"• Uses {features['exclamation_count']} exclamation marks")
        
        if not explanations:
            explanations.append("• No significant suspicious patterns detected")
        
        return "\n".join(explanations)

# Initialize detector
detector = PhishingDetector()

# Sample phishing emails for testing
sample_emails = {
    "Suspicious Banking Email": """URGENT: Your Bank Account Has Been Suspended!

Dear Customer,

Your account has been temporarily suspended due to suspicious activity. 
You must verify your identity IMMEDIATELY to avoid permanent closure.

Click here to verify: http://bit.ly/verify-account-now

Act fast! This link expires in 24 hours.

Best regards,
Security Team""",
    
    "Lottery Scam": """CONGRATULATIONS! YOU'VE WON $1,000,000!!!

You have been selected as the winner of our international lottery!
To claim your prize, please provide:
- Full name
- Address  
- Phone number
- Bank details

Reply immediately as this offer expires soon!""",
    
    "Legitimate Email": """Hi,

Hope you're doing well. Just wanted to follow up on our meeting yesterday.
Could you please send me the project timeline when you get a chance?

Thanks,
John"""
}

def analyze_with_sample(sample_choice, email_text, subject, sender):
    if sample_choice != "Custom Email":
        email_text = sample_emails[sample_choice]
    
    return detector.analyze_email(email_text, subject, sender)

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="AI Phishing Email Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🛡️ AI Phishing Email Detector
        
        **Protect yourself from phishing attacks using advanced AI analysis!**
        
        This tool analyzes emails for suspicious patterns, phishing keywords, and social engineering tactics.
        Built with Hugging Face Transformers and deployed on Hugging Face Spaces.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📧 Email Analysis")
                
                sample_dropdown = gr.Dropdown(
                    choices=["Custom Email"] + list(sample_emails.keys()),
                    value="Custom Email",
                    label="Try Sample Emails or Enter Custom",
                    interactive=True
                )
                
                email_input = gr.Textbox(
                    lines=10,
                    placeholder="Paste the email content here...",
                    label="Email Content",
                    interactive=True
                )
                
                with gr.Row():
                    subject_input = gr.Textbox(
                        placeholder="Email subject (optional)",
                        label="Subject Line",
                        interactive=True
                    )
                    sender_input = gr.Textbox(
                        placeholder="Sender email (optional)",
                        label="Sender",
                        interactive=True
                    )
                
                analyze_btn = gr.Button("🔍 Analyze Email", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Results")
                
                risk_output = gr.Textbox(label="Risk Level", interactive=False)
                score_output = gr.Number(label="Risk Score (%)", interactive=False)
                
                gr.Markdown("### 📋 Detailed Analysis")
                explanation_output = gr.Textbox(
                    lines=6,
                    label="Risk Factors",
                    interactive=False
                )
        
        gr.Markdown("""
        ### 🎯 How It Works
        
        This AI detector analyzes multiple factors:
        - **Phishing Keywords**: Common phishing terminology
        - **Urgency Tactics**: Time pressure and urgent language  
        - **URL Analysis**: Suspicious links and domains
        - **Content Patterns**: Unusual formatting and grammar
        - **AI Sentiment**: Advanced language understanding
        
        ### 🔧 Technology Stack
        - **AI Model**: DistilBERT (Hugging Face Transformers)
        - **Interface**: Gradio
        - **Analysis**: Multi-layer feature extraction
        - **Deployment**: Hugging Face Spaces
        
        ### ⚠️ Disclaimer
        This is a demo tool for educational purposes. Always use multiple verification methods for important emails.
        
        ---
        **Created by**: [Your Name] | **GitHub**: [Your GitHub Link] | **Demo for Hackathon**
        """)
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_with_sample,
            inputs=[sample_dropdown, email_input, subject_input, sender_input],
            outputs=[risk_output, score_output, explanation_output]
        )
        
        sample_dropdown.change(
            fn=lambda x: sample_emails.get(x, "") if x != "Custom Email" else "",
            inputs=[sample_dropdown],
            outputs=[email_input]
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()