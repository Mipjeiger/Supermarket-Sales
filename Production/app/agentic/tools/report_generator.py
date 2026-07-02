import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
from pathlib import Path
import markdown
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

"""
Report Generator Tool for Agentic System
Generates detailed investigation reports.
"""

logger = logging.getLogger(__name__)

class ReportGeneratorTool:
    """
    Tool for generating investigation reports:
    - Fraud investigation reports
    - Security analysis reports
    - Performance reports
    - Executive summaries
    """

    def __init__(self):
        self.report_templates = self._load_templates()
        self.report_cache = {}
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)

    def _load_templates(self) -> Dict:
        """Load report templates."""
        return {
            'fraud_investigation': {
                'html': self._get_fraud_investigation_template(),
                'markdown': self._get_fraud_investigation_md_template(),
                'json': self._get_fraud_investigation_json_template()
            },
            'security_analysis': {
                'html': self._get_security_analysis_template(),
                'markdown': self._get_security_analysis_md_template()
            },
            'performance': {
                'html': self._get_performance_template()
            },
            'executive_summary': {
                'html': self._get_executive_summary_template()
            }
        }
    
    def run(self, data: str) -> Dict:
        """
        Main entry point for the tool.
        
        Args:
            data: JSON string or dict with report data
            
        Returns:
            Dict containing generated report
        """
        try:
            # Parse data if string
            if isinstance(data, str):
                data = json.loads(data)
            
            report_type = data.get('report_type', 'fraud_investigation')
            report_data = data.get('data', {})
            format = data.get('format', 'json')
            
            # Generate report
            report = self._generate_report(report_type, report_data, format)
            
            # Save report
            report_id = self._save_report(report, report_type)
            
            return {
                'report_id': report_id,
                'report_type': report_type,
                'format': format,
                'content': report,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {str(e)}")
            return {'error': str(e)}
        
    def _generate_report(self, report_type: str, data: Dict, format: str) -> Any:
        """
        Generate report in specified format.
        """
        if report_type == 'fraud_investigation':
            return self._generate_fraud_investigation_report(data, format)
        elif report_type == 'security_analysis':
            return self._generate_security_analysis_report(data, format)
        elif report_type == 'performance':
            return self._generate_performance_report(data, format)
        elif report_type == 'executive_summary':
            return self._generate_executive_summary(data, format)
        else:
            return {'error': f'Unknown report type: {report_type}'}
        
    def _generate_fraud_investigation_report(self, data: Dict, format: str) -> Dict:
        """Generate fraud investigation report"""

        # Extract data
        investigation = data.get('investigation', {})
        findings = data.get('findings', [])
        risk_assessment = data.get('risk_assessment', {})
        recommendations = data.get('recommendations', [])

        # Generate visualizations
        charts = self._generate_charts(data)

        report = {
            'title': 'Fraud Investigation Report',
            'investigation_id': investigation.get('investigation_id', ''),
            'timestamp': datetime.now().isoformat(),
            'executive_summary': self._generate_executive_summary_text(data),
            'findings': findings,
            'risk_assessment': risk_assessment,
            'recommendations': recommendations,
            'charts': charts,
            'evidence': self._format_evidence(data.get('evidence', [])),
            'action_items': self._generate_action_items(recommendations)
        }

        if format == 'html':
            return self._render_html_report(report, 'fraud_investigation')
        elif format == 'markdown':
            return self._render_markdown_report(report, 'fraud_investigation')
        else:  # json
            return report
        
    def _generate_security_analysis_report(self, data: Dict, format: str) -> Dict:
        """
        Generate security analysis report.
        """
        report = {
            'title': 'Security Analysis Report',
            'analysis_id': data.get('analysis_id', ''),
            'timestamp': datetime.now().isoformat(),
            'threats_identified': data.get('threats', []),
            'vulnerabilities': data.get('vulnerabilities', []),
            'risk_level': data.get('risk_level', 'low'),
            'recommendations': data.get('recommendations', []),
            'security_score': data.get('security_score', 0),
            'action_items': self._generate_action_items(data.get('recommendations', []))
        }
        
        if format == 'html':
            return self._render_html_report(report, 'security_analysis')
        elif format == 'markdown':
            return self._render_markdown_report(report, 'security_analysis')
        else:
            return report
        
    def _generate_performance_report(self, data: Dict, format: str) -> Dict:
        """
        Generate performance report.
        """
        report = {
            'title': 'System Performance Report',
            'timestamp': datetime.now().isoformat(),
            'metrics': data.get('metrics', {}),
            'performance_summary': self._generate_performance_summary(data),
            'bottlenecks': self._identify_bottlenecks(data),
            'recommendations': data.get('recommendations', [])
        }
        
        if format == 'html':
            return self._render_html_report(report, 'performance')
        else:
            return report
    
    def _generate_executive_summary(self, data: Dict, format: str) -> Dict:
        """
        Generate executive summary.
        """
        report = {
            'title': 'Executive Summary',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'key_findings': data.get('key_findings', []),
            'metrics': data.get('metrics', {}),
            'critical_issues': data.get('critical_issues', []),
            'recommendations': data.get('recommendations', []),
            'next_steps': data.get('next_steps', [])
        }
        
        if format == 'html':
            return self._render_html_report(report, 'executive_summary')
        else:
            return report
        
    def _generate_charts(self, data: Dict) -> List[Dict]:
        """
        Generate charts for the report.
        """
        charts = []
        
        try:
            # Risk distribution chart
            if 'risk_assessment' in data:
                risk_data = data['risk_assessment']
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Create risk distribution
                categories = ['Low', 'Medium', 'High', 'Critical']
                values = [
                    risk_data.get('low', 0),
                    risk_data.get('medium', 0),
                    risk_data.get('high', 0),
                    risk_data.get('critical', 0)
                ]
                
                ax.bar(categories, values, color=['green', 'yellow', 'orange', 'red'])
                ax.set_title('Risk Distribution')
                ax.set_ylabel('Count')
                
                # Convert to base64
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='png', bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                charts.append({
                    'title': 'Risk Distribution',
                    'type': 'bar',
                    'data': img_base64
                })
                plt.close()
            
            # Trend chart
            if 'findings' in data and data['findings']:
                fig, ax = plt.subplots(figsize=(10, 6))
                # Plot trends
                ax.plot(range(len(data['findings'])), 
                       [f.get('score', 0) for f in data['findings']],
                       marker='o')
                ax.set_title('Findings Trend')
                ax.set_xlabel('Finding Index')
                ax.set_ylabel('Score')
                
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='png', bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                charts.append({
                    'title': 'Findings Trend',
                    'type': 'line',
                    'data': img_base64
                })
                plt.close()
                
        except Exception as e:
            logger.error(f"❌ Chart generation failed: {str(e)}")
        
        return charts
    
    def _generate_action_items(self, recommendations: List) -> List[Dict]:
        """
        Generate action items from recommendations.
        """
        action_items = []
        
        for i, rec in enumerate(recommendations):
            action_items.append({
                'id': f'ACT-{i+1:03d}',
                'description': rec if isinstance(rec, str) else rec.get('description', ''),
                'priority': rec.get('priority', 'medium') if isinstance(rec, dict) else 'medium',
                'status': 'pending',
                'assigned_to': rec.get('assigned_to', 'TBD') if isinstance(rec, dict) else 'TBD'
            })
        
        return action_items
    
    def _generate_performance_summary(self, data: Dict) -> str:
        """
        Generate performance summary text.
        """
        metrics = data.get('metrics', {})
        
        summary = f"""
        Overall Performance: {metrics.get('overall_score', 0):.1f}%
        Response Time: {metrics.get('avg_response_time', 0):.2f}ms
        Throughput: {metrics.get('throughput', 0)} req/s
        Error Rate: {metrics.get('error_rate', 0):.2f}%
        """
        
        return summary.strip()
    
    def _identify_bottlenecks(self, data: Dict) -> List[Dict]:
        """
        Identify system bottlenecks.
        """
        bottlenecks = []
        metrics = data.get('metrics', {})
        
        if metrics.get('avg_response_time', 0) > 1000:
            bottlenecks.append({
                'component': 'API',
                'issue': 'High latency',
                'severity': 'high'
            })
        
        if metrics.get('error_rate', 0) > 5:
            bottlenecks.append({
                'component': 'Database',
                'issue': 'High error rate',
                'severity': 'critical'
            })
        
        return bottlenecks
    
    def _render_html_report(self, report: Dict, template_type: str) -> str:
        """
        Render report as HTML.
        """
        template = self.report_templates.get(template_type, {}).get('html', '')
        if not template:
            return self._generate_default_html(report)
        
        # Render template with data
        html_template = Template(template)
        return html_template.render(**report)
    
    def _render_markdown_report(self, report: Dict, template_type: str) -> str:
        """
        Render report as Markdown.
        """
        template = self.report_templates.get(template_type, {}).get('markdown', '')
        if not template:
            return self._generate_default_markdown(report)
        
        md_template = Template(template)
        return md_template.render(**report)
    
    def _save_report(self, report: Dict, report_type: str) -> str:
        """
        Save report to file.
        """
        report_id = f"{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine format
        if isinstance(report, dict) and 'content' in report:
            content = report['content']
        else:
            content = report
        
        # Save based on type
        if isinstance(content, str):
            # Save as HTML/Markdown
            extension = 'html' if '<html' in content else 'md'
            file_path = self.output_dir / f"{report_id}.{extension}"
            with open(file_path, 'w') as f:
                f.write(content)
        else:
            # Save as JSON
            file_path = self.output_dir / f"{report_id}.json"
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2, default=str)
        
        logger.info(f"📄 Report saved: {file_path}")
        return report_id
    
    def _generate_executive_summary_text(self, data: Dict) -> str:
        """Generate executive summary text."""
        risk_level = data.get('risk_assessment', {}).get('level', 'unknown')
        findings_count = len(data.get('findings', []))
        critical_findings = len([f for f in data.get('findings', []) if f.get('severity') == 'critical'])
        
        summary = f"""
        This investigation analyzed {findings_count} findings with {critical_findings} critical issues.
        Overall risk level is {risk_level}.
        {'Immediate action required' if risk_level in ['high', 'critical'] else 'Continued monitoring recommended'}.
        """
        return summary.strip()
    
    def _format_evidence(self, evidence: List) -> List:
        """Format evidence for the report."""
        return evidence
    
    def _get_fraud_investigation_template(self) -> str:
        """HTML template for fraud investigation."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #2c3e50; color: white; padding: 20px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .critical { background: #ff6b6b; color: white; padding: 5px 10px; border-radius: 3px; }
                .high { background: #ffa94d; padding: 5px 10px; border-radius: 3px; }
                .medium { background: #ffd93d; padding: 5px 10px; border-radius: 3px; }
                .low { background: #6bcb77; padding: 5px 10px; border-radius: 3px; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                .chart { margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <p>Investigation ID: {{ investigation_id }}</p>
                <p>Date: {{ timestamp }}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>{{ executive_summary }}</p>
            </div>
            
            <div class="section">
                <h2>Risk Assessment</h2>
                <p>Risk Level: <span class="{{ risk_assessment.level }}">{{ risk_assessment.level.upper() }}</span></p>
                <p>Risk Score: {{ risk_assessment.score }}</p>
            </div>
            
            <div class="section">
                <h2>Findings</h2>
                {% for finding in findings %}
                <div style="margin: 10px 0; padding: 10px; background: #f8f9fa;">
                    <p><strong>Finding {{ loop.index }}:</strong> {{ finding.description }}</p>
                    <p>Severity: <span class="{{ finding.severity }}">{{ finding.severity.upper() }}</span></p>
                    <p>Evidence: {{ finding.evidence }}</p>
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>Charts</h2>
                {% for chart in charts %}
                <div class="chart">
                    <h3>{{ chart.title }}</h3>
                    <img src="data:image/png;base64,{{ chart.data }}" alt="{{ chart.title }}">
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <div class="section">
                <h2>Action Items</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Description</th>
                        <th>Priority</th>
                        <th>Status</th>
                        <th>Assigned To</th>
                    </tr>
                    {% for item in action_items %}
                    <tr>
                        <td>{{ item.id }}</td>
                        <td>{{ item.description }}</td>
                        <td>{{ item.priority }}</td>
                        <td>{{ item.status }}</td>
                        <td>{{ item.assigned_to }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        """
    
    def _get_fraud_investigation_md_template(self) -> str:
        """Markdown template for fraud investigation."""
        return """
        # {{ title }}
        
        **Investigation ID:** {{ investigation_id }}
        **Date:** {{ timestamp }}
        
        ## Executive Summary
        {{ executive_summary }}
        
        ## Risk Assessment
        - **Risk Level:** {{ risk_assessment.level.upper() }}
        - **Risk Score:** {{ risk_assessment.score }}
        
        ## Findings
        {% for finding in findings %}
        ### Finding {{ loop.index }}
        - **Description:** {{ finding.description }}
        - **Severity:** {{ finding.severity.upper() }}
        - **Evidence:** {{ finding.evidence }}
        
        {% endfor %}
        ## Recommendations
        {% for rec in recommendations %}
        - {{ rec }}
        {% endfor %}
        
        ## Action Items
        | ID | Description | Priority | Status | Assigned To |
        |-----|-------------|----------|---------|-------------|
        {% for item in action_items %}
        | {{ item.id }} | {{ item.description }} | {{ item.priority }} | {{ item.status }} | {{ item.assigned_to }} |
        {% endfor %}
        """
    
    def _get_security_analysis_template(self) -> str:
        """HTML template for security analysis."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #2c3e50; color: white; padding: 20px; }
                .threat { background: #ff6b6b; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .vulnerability { background: #ffa94d; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <p>Analysis ID: {{ analysis_id }}</p>
                <p>Date: {{ timestamp }}</p>
            </div>
            
            <div>
                <h2>Security Score</h2>
                <div class="metric">
                    <h3>{{ security_score }}%</h3>
                    <p>Overall Security Score</p>
                </div>
                <div class="metric">
                    <h3>{{ risk_level.upper() }}</h3>
                    <p>Risk Level</p>
                </div>
            </div>
            
            <div>
                <h2>Threats Identified</h2>
                {% for threat in threats_identified %}
                <div class="threat">
                    <h3>{{ threat.name }}</h3>
                    <p>{{ threat.description }}</p>
                    <p>Severity: {{ threat.severity }}</p>
                </div>
                {% endfor %}
            </div>
            
            <div>
                <h2>Vulnerabilities</h2>
                {% for vuln in vulnerabilities %}
                <div class="vulnerability">
                    <h3>{{ vuln.name }}</h3>
                    <p>{{ vuln.description }}</p>
                    <p>Risk: {{ vuln.risk }}</p>
                </div>
                {% endfor %}
            </div>
            
            <div>
                <h2>Recommendations</h2>
                <ul>
                {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            </div>
        </body>
        </html>
        """
    
    def _get_security_analysis_md_template(self) -> str:
        """Markdown template for security analysis."""
        return """
        # {{ title }}
        
        **Analysis ID:** {{ analysis_id }}
        **Date:** {{ timestamp }}
        
        ## Security Score
        **Overall Score:** {{ security_score }}%
        **Risk Level:** {{ risk_level.upper() }}
        
        ## Threats Identified
        {% for threat in threats_identified %}
        ### {{ threat.name }}
        - **Description:** {{ threat.description }}
        - **Severity:** {{ threat.severity }}
        {% endfor %}
        
        ## Vulnerabilities
        {% for vuln in vulnerabilities %}
        ### {{ vuln.name }}
        - **Description:** {{ vuln.description }}
        - **Risk:** {{ vuln.risk }}
        {% endfor %}
        
        ## Recommendations
        {% for rec in recommendations %}
        - {{ rec }}
        {% endfor %}
        """
    
    def _get_performance_template(self) -> str:
        """HTML template for performance report."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .metric { display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
                .bottleneck { background: #ff6b6b; padding: 10px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            <p>Date: {{ timestamp }}</p>
            
            <h2>Performance Summary</h2>
            <p>{{ performance_summary }}</p>
            
            <div>
                <h2>Metrics</h2>
                {% for key, value in metrics.items() %}
                <div class="metric">
                    <h4>{{ key }}</h4>
                    <p>{{ value }}</p>
                </div>
                {% endfor %}
            </div>
            
            <div>
                <h2>Bottlenecks</h2>
                {% for bottleneck in bottlenecks %}
                <div class="bottleneck">
                    <h3>{{ bottleneck.component }}</h3>
                    <p>{{ bottleneck.issue }}</p>
                    <p>Severity: {{ bottleneck.severity }}</p>
                </div>
                {% endfor %}
            </div>
            
            <div>
                <h2>Recommendations</h2>
                <ul>
                {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            </div>
        </body>
        </html>
        """
    
    def _get_executive_summary_template(self) -> str:
        """HTML template for executive summary."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #2c3e50; color: white; padding: 20px; }
                .key-finding { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }
                .critical { border-left-color: #e74c3c; }
                .metric-box { display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ title }}</h1>
                <p>Date: {{ date }}</p>
            </div>
            
            <div>
                <h2>Key Metrics</h2>
                {% for key, value in metrics.items() %}
                <div class="metric-box">
                    <h4>{{ key }}</h4>
                    <p>{{ value }}</p>
                </div>
                {% endfor %}
            </div>
            
            <div>
                <h2>Key Findings</h2>
                {% for finding in key_findings %}
                <div class="key-finding {% if finding.critical %}critical{% endif %}">
                    <h3>{{ finding.title }}</h3>
                    <p>{{ finding.description }}</p>
                    <p>Impact: {{ finding.impact }}</p>
                </div>
                {% endfor %}
            </div>
            
            <div>
                <h2>Critical Issues</h2>
                <ul>
                {% for issue in critical_issues %}
                    <li>{{ issue }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <div>
                <h2>Recommendations</h2>
                <ul>
                {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            </div>
            
            <div>
                <h2>Next Steps</h2>
                <ol>
                {% for step in next_steps %}
                    <li>{{ step }}</li>
                {% endfor %}
                </ol>
            </div>
        </body>
        </html>
        """
    
    def _get_fraud_investigation_json_template(self) -> Dict:
        """JSON template for fraud investigation."""
        return {
            'title': 'Fraud Investigation Report',
            'sections': [
                'investigation_id',
                'timestamp',
                'executive_summary',
                'findings',
                'risk_assessment',
                'recommendations',
                'action_items'
            ]
        }
    
    def _generate_default_html(self, report: Dict) -> str:
        """Generate default HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Investigation Report</title></head>
        <body>
            <h1>Investigation Report</h1>
            <p>Generated: {datetime.now().isoformat()}</p>
            <pre>{json.dumps(report, indent=2, default=str)}</pre>
        </body>
        </html>
        """
        return html
    
    def _generate_default_markdown(self, report: Dict) -> str:
        """Generate default Markdown report."""
        md = f"""
        # Investigation Report
        **Generated:** {datetime.now().isoformat()}
        
        ```json"""
        {json.dumps(report, indent=2, default=str)}