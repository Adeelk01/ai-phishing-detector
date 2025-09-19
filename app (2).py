import gradio as gr
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import warnings
import json
from datetime import datetime
import os
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
        
        # Initialize history file
        self.history_file = "analysis_history.json"
        
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
    
    def save_analysis_history(self, email_text, subject, sender, risk_level, risk_score, explanation):
        """Save analysis to history file"""
        try:
            # Create analysis record
            analysis_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'email_preview': email_text[:100] + "..." if len(email_text) > 100 else email_text,
                'subject': subject,
                'sender': sender,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'explanation': explanation
            }
            
            # Load existing history
            history = self.load_analysis_history()
            
            # Add new record at the beginning
            history.insert(0, analysis_record)
            
            # Keep only last 100 records
            history = history[:100]
            
            # Save back to file
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            return f"✅ Analysis saved to history ({len(history)} total records)"
            
        except Exception as e:
            return f"⚠️ Could not save to history: {str(e)}"
    
    def load_analysis_history(self):
        """Load analysis history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return []
        except:
            return []
    
    def get_history_stats(self):
        """Get statistics from analysis history"""
        history = self.load_analysis_history()
        
        if not history:
            return "No analysis history available yet."
        
        total_analyses = len(history)
        high_risk = len([h for h in history if 'HIGH RISK' in h['risk_level']])
        medium_risk = len([h for h in history if 'MEDIUM RISK' in h['risk_level']])
        low_risk = len([h for h in history if 'LOW RISK' in h['risk_level']])
        
        avg_risk_score = np.mean([h['risk_score'] for h in history])
        
        stats = f"""
📊 **ANALYSIS STATISTICS**

📧 Total Emails Analyzed: {total_analyses}
🔴 High Risk Detected: {high_risk} ({high_risk/total_analyses*100:.1f}%)
🟡 Medium Risk Detected: {medium_risk} ({medium_risk/total_analyses*100:.1f}%)
🟢 Low Risk (Safe): {low_risk} ({low_risk/total_analyses*100:.1f}%)

📈 Average Risk Score: {avg_risk_score:.1f}%
🛡️ Threats Blocked: {high_risk + medium_risk}
        """
        
        return stats
    
    def get_recent_history(self, limit=10):
        """Get recent analysis history"""
        history = self.load_analysis_history()
        
        if not history:
            return "No recent analyses available."
        
        recent = history[:limit]
        
        history_text = "🕐 **RECENT ANALYSES:**\n\n"
        
        for i, record in enumerate(recent, 1):
            risk_emoji = "🔴" if "HIGH" in record['risk_level'] else "🟡" if "MEDIUM" in record['risk_level'] else "🟢"
            history_text += f"{i}. {risk_emoji} **{record['timestamp']}**\n"
            history_text += f"   Subject: {record['subject'][:50]}...\n"
            history_text += f"   Risk: {record['risk_level']}\n"
            history_text += f"   Score: {record['risk_score']}%\n\n"
        
        return history_text
    
    def analyze_email(self, email_text, subject="", sender=""):
        """Main analysis function"""
        if not email_text.strip():
            return "Please enter email content to analyze.", 0.0, "", "No analysis performed.", ""
        
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
        
        # Save to history
        save_status = self.save_analysis_history(email_text, subject, sender, risk_level, round(risk_score * 100, 1), explanation)
        
        return risk_level, round(risk_score * 100, 1), explanation, save_status
    
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

def get_stats():
    return detector.get_history_stats()

def get_history():
    return detector.get_recent_history()

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="AI Phishing Email Detector", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🛡️ AI Phishing Email Detector
        
        **Protect yourself from phishing attacks using advanced AI analysis!**
        
        This tool analyzes emails for suspicious patterns, phishing keywords, and social engineering tactics.
        Built with Hugging Face Transformers and now with **Analysis History** feature!
        """)
        
        with gr.Tabs():
            # Main Analysis Tab
            with gr.TabItem("📧 Email Analysis"):
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
                        
                        # NEW: Save status display
                        save_status = gr.Textbox(
                            label="Save Status",
                            interactive=False,
                            visible=True
                        )
            
            # NEW: History Tab
            with gr.TabItem("📊 Analysis History"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📈 Your Analysis Statistics")
                        
                        stats_btn = gr.Button("📊 Refresh Statistics", variant="secondary")
                        stats_output = gr.Textbox(
                            lines=15,
                            label="Statistics",
                            interactive=False
                        )
                        
                    with gr.Column():
                        gr.Markdown("### 🕐 Recent Analyses")
                        
                        history_btn = gr.Button("🕐 Refresh History", variant="secondary")
                        history_output = gr.Textbox(
                            lines=15,
                            label="Recent History",
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
        
        ### 🆕 New Features
        - **Analysis History**: All your analyses are automatically saved
        - **Statistics Dashboard**: Track your email security over time
        - **Recent Activity**: View your last 10 analyses
        
        ### ⚠️ Disclaimer
        This is a demo tool for educational purposes. Always use multiple verification methods for important emails.
        
        ---
        **Created by**: [Your Name] | **GitHub**: [Your GitHub Link] | **Enhanced Version with History**
        """)
        
        # Event handlers for main analysis
        analyze_btn.click(
            fn=analyze_with_sample,
            inputs=[sample_dropdown, email_input, subject_input, sender_input],
            outputs=[risk_output, score_output, explanation_output, save_status]
        )
        
        sample_dropdown.change(
            fn=lambda x: sample_emails.get(x, "") if x != "Custom Email" else "",
            inputs=[sample_dropdown],
            outputs=[email_input]
        )
        
        # Event handlers for history
        stats_btn.click(
            fn=get_stats,
            outputs=[stats_output]
        )
        
        history_btn.click(
            fn=get_history,
            outputs=[history_output]
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,  # Creates public link
        server_name="0.0.0.0",  # For Colab
        server_port=7860,
        show_error=True
    )
