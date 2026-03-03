"""
HTML Report Generator
=====================
Production-grade HTML report generation for integration test results.

Generates comprehensive, visually appealing reports with:
- Test summary and statistics
- Pass/fail status for each test suite
- Performance metrics and charts
- Error details and stack traces
- Recommendations and action items
- Timestamp and system information
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    status: TestStatus
    duration: float  # seconds
    message: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TestSuite:
    """Test suite results."""
    name: str
    description: str
    tests: List[TestResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def passed_count(self) -> int:
        """Count of passed tests."""
        return sum(1 for t in self.tests if t.status == TestStatus.PASSED)

    @property
    def failed_count(self) -> int:
        """Count of failed tests."""
        return sum(1 for t in self.tests if t.status == TestStatus.FAILED)

    @property
    def skipped_count(self) -> int:
        """Count of skipped tests."""
        return sum(1 for t in self.tests if t.status == TestStatus.SKIPPED)

    @property
    def error_count(self) -> int:
        """Count of tests with errors."""
        return sum(1 for t in self.tests if t.status == TestStatus.ERROR)

    @property
    def total_count(self) -> int:
        """Total count of tests."""
        return len(self.tests)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.passed_count / self.total_count) * 100.0


@dataclass
class PerformanceMetric:
    """Performance metric."""
    name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    passed: bool = True
    description: str = ""


@dataclass
class SystemInfo:
    """System information."""
    python_version: str
    platform: str
    hostname: str
    test_timestamp: str
    total_duration: float


@dataclass
class IntegrationReport:
    """Complete integration test report."""
    title: str
    system_info: SystemInfo
    test_suites: List[TestSuite] = field(default_factory=list)
    performance_metrics: List[PerformanceMetric] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        """Total count of all tests."""
        return sum(suite.total_count for suite in self.test_suites)

    @property
    def total_passed(self) -> int:
        """Total passed tests."""
        return sum(suite.passed_count for suite in self.test_suites)

    @property
    def total_failed(self) -> int:
        """Total failed tests."""
        return sum(suite.failed_count for suite in self.test_suites)

    @property
    def total_errors(self) -> int:
        """Total errors."""
        return sum(suite.error_count for suite in self.test_suites)

    @property
    def overall_success_rate(self) -> float:
        """Overall success rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.total_passed / self.total_tests) * 100.0


class HTMLReportGenerator:
    """
    Generates HTML reports for integration test results.

    Features:
    - Beautiful, responsive HTML
    - Color-coded status indicators
    - Collapsible sections
    - Performance charts
    - Search and filtering
    - Exportable to PDF
    """

    def __init__(self):
        """Initialize HTML report generator."""
        self.template = self._get_template()

    def generate_report(
        self,
        report: IntegrationReport,
        output_path: Path
    ) -> Path:
        """
        Generate HTML report.

        Args:
            report: Integration test report data
            output_path: Output HTML file path

        Returns:
            Path to generated HTML file
        """
        logger.info(f"Generating HTML report: {output_path}")

        # Generate HTML content
        html = self._render_template(report)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        logger.info(f"HTML report generated successfully: {output_path}")

        return output_path

    def _render_template(self, report: IntegrationReport) -> str:
        """Render HTML template with report data."""
        # Generate sections
        summary_html = self._generate_summary_section(report)
        suites_html = self._generate_suites_section(report)
        performance_html = self._generate_performance_section(report)
        issues_html = self._generate_issues_section(report)

        # Replace placeholders
        html = self.template
        html = html.replace("{{TITLE}}", report.title)
        html = html.replace("{{TIMESTAMP}}", report.system_info.test_timestamp)
        html = html.replace("{{SUMMARY}}", summary_html)
        html = html.replace("{{TEST_SUITES}}", suites_html)
        html = html.replace("{{PERFORMANCE}}", performance_html)
        html = html.replace("{{ISSUES}}", issues_html)

        return html

    def _generate_summary_section(self, report: IntegrationReport) -> str:
        """Generate summary section HTML."""
        status_class = "success" if report.overall_success_rate >= 80.0 else "warning" if report.overall_success_rate >= 60.0 else "danger"

        html = f"""
        <div class="summary-section">
            <div class="summary-card">
                <h2>Test Summary</h2>
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-number">{report.total_tests}</div>
                        <div class="summary-label">Total Tests</div>
                    </div>
                    <div class="summary-item success">
                        <div class="summary-number">{report.total_passed}</div>
                        <div class="summary-label">Passed</div>
                    </div>
                    <div class="summary-item danger">
                        <div class="summary-number">{report.total_failed}</div>
                        <div class="summary-label">Failed</div>
                    </div>
                    <div class="summary-item warning">
                        <div class="summary-number">{report.total_errors}</div>
                        <div class="summary-label">Errors</div>
                    </div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill {status_class}" style="width: {report.overall_success_rate}%">
                        {report.overall_success_rate:.1f}%
                    </div>
                </div>
            </div>
            
            <div class="summary-card">
                <h2>System Information</h2>
                <table class="info-table">
                    <tr>
                        <td><strong>Python Version:</strong></td>
                        <td>{report.system_info.python_version}</td>
                    </tr>
                    <tr>
                        <td><strong>Platform:</strong></td>
                        <td>{report.system_info.platform}</td>
                    </tr>
                    <tr>
                        <td><strong>Hostname:</strong></td>
                        <td>{report.system_info.hostname}</td>
                    </tr>
                    <tr>
                        <td><strong>Total Duration:</strong></td>
                        <td>{report.system_info.total_duration:.2f}s</td>
                    </tr>
                </table>
            </div>
        </div>
        """

        return html

    def _generate_suites_section(self, report: IntegrationReport) -> str:
        """Generate test suites section HTML."""
        html = '<div class="suites-section">'

        for suite in report.test_suites:
            status_class = "success" if suite.failed_count == 0 else "danger"
            status_icon = "✓" if suite.failed_count == 0 else "✗"

            html += f"""
            <div class="suite-card">
                <div class="suite-header {status_class}">
                    <h3><span class="status-icon">{status_icon}</span> {suite.name}</h3>
                    <div class="suite-stats">
                        <span class="stat success">{suite.passed_count} passed</span>
                        <span class="stat danger">{suite.failed_count} failed</span>
                        <span class="stat">{suite.total_duration:.2f}s</span>
                    </div>
                </div>
                <div class="suite-description">{suite.description}</div>
                <div class="test-list">
            """

            for test in suite.tests:
                test_status_class = test.status.value
                test_status_icon = "✓" if test.status == TestStatus.PASSED else "✗" if test.status == TestStatus.FAILED else "⚠"

                html += f"""
                <div class="test-item {test_status_class}">
                    <div class="test-header">
                        <span class="test-status-icon">{test_status_icon}</span>
                        <span class="test-name">{test.name}</span>
                        <span class="test-duration">{test.duration:.3f}s</span>
                    </div>
                """

                if test.message:
                    html += f'<div class="test-message">{test.message}</div>'

                if test.error:
                    html += f'''
                    <div class="test-error">
                        <strong>Error:</strong> {test.error}
                    </div>
                    '''

                if test.traceback:
                    html += f'''
                    <details class="test-traceback">
                        <summary>Traceback</summary>
                        <pre>{test.traceback}</pre>
                    </details>
                    '''

                html += '</div>'

            html += """
                </div>
            </div>
            """

        html += '</div>'
        return html

    def _generate_performance_section(self, report: IntegrationReport) -> str:
        """Generate performance metrics section HTML."""
        if not report.performance_metrics:
            return '<div class="performance-section"><p>No performance metrics available.</p></div>'

        html = '<div class="performance-section"><h2>Performance Metrics</h2><div class="metrics-grid">'

        for metric in report.performance_metrics:
            status_class = "success" if metric.passed else "danger"
            status_icon = "✓" if metric.passed else "✗"

            threshold_text = ""
            if metric.threshold is not None:
                threshold_text = f" (threshold: {metric.threshold} {metric.unit})"

            html += f"""
            <div class="metric-card {status_class}">
                <div class="metric-header">
                    <span class="metric-icon">{status_icon}</span>
                    <span class="metric-name">{metric.name}</span>
                </div>
                <div class="metric-value">{metric.value:.2f} {metric.unit}</div>
                <div class="metric-threshold">{threshold_text}</div>
                <div class="metric-description">{metric.description}</div>
            </div>
            """

        html += '</div></div>'
        return html

    def _generate_issues_section(self, report: IntegrationReport) -> str:
        """Generate issues and recommendations section HTML."""
        html = '<div class="issues-section">'

        if report.issues:
            html += '<div class="issues-card"><h2>Issues Found</h2><ul class="issues-list">'
            for issue in report.issues:
                html += f'<li class="issue-item danger">{issue}</li>'
            html += '</ul></div>'

        if report.recommendations:
            html += '<div class="recommendations-card"><h2>Recommendations</h2><ul class="recommendations-list">'
            for rec in report.recommendations:
                html += f'<li class="recommendation-item">{rec}</li>'
            html += '</ul></div>'

        html += '</div>'
        return html

    def _get_template(self) -> str:
        """Get HTML template."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header .timestamp {
            opacity: 0.9;
            font-size: 0.95rem;
        }
        
        .content {
            padding: 40px;
        }
        
        .summary-section {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .summary-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 25px;
        }
        
        .summary-card h2 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #667eea;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .summary-item {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
        }
        
        .summary-item.success {
            border-color: #4caf50;
        }
        
        .summary-item.danger {
            border-color: #f44336;
        }
        
        .summary-item.warning {
            border-color: #ff9800;
        }
        
        .summary-number {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .summary-label {
            font-size: 0.85rem;
            color: #666;
        }
        
        .progress-bar {
            width: 100%;
            height: 40px;
            background: #e0e0e0;
            border-radius: 20px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.3s ease;
        }
        
        .progress-fill.success {
            background: linear-gradient(90deg, #4caf50, #8bc34a);
        }
        
        .progress-fill.warning {
            background: linear-gradient(90deg, #ff9800, #ffc107);
        }
        
        .progress-fill.danger {
            background: linear-gradient(90deg, #f44336, #e91e63);
        }
        
        .info-table {
            width: 100%;
        }
        
        .info-table td {
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .info-table tr:last-child td {
            border-bottom: none;
        }
        
        .suite-card {
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        
        .suite-header {
            padding: 20px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .suite-header.success {
            background: linear-gradient(90deg, #4caf50, #8bc34a);
        }
        
        .suite-header.danger {
            background: linear-gradient(90deg, #f44336, #e91e63);
        }
        
        .suite-header h3 {
            font-size: 1.3rem;
        }
        
        .suite-stats {
            display: flex;
            gap: 15px;
        }
        
        .suite-stats .stat {
            padding: 5px 10px;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
            font-size: 0.9rem;
        }
        
        .suite-description {
            padding: 15px 20px;
            background: white;
            border-bottom: 1px solid #e0e0e0;
            color: #666;
        }
        
        .test-list {
            padding: 10px;
        }
        
        .test-item {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
            border-left: 4px solid #e0e0e0;
        }
        
        .test-item.passed {
            border-left-color: #4caf50;
        }
        
        .test-item.failed {
            border-left-color: #f44336;
        }
        
        .test-header {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .test-status-icon {
            font-size: 1.2rem;
        }
        
        .test-name {
            flex: 1;
            font-weight: 500;
        }
        
        .test-duration {
            color: #666;
            font-size: 0.9rem;
        }
        
        .test-message {
            margin-top: 10px;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        
        .test-error {
            margin-top: 10px;
            padding: 10px;
            background: #ffebee;
            border-radius: 4px;
            color: #c62828;
            font-size: 0.9rem;
        }
        
        .test-traceback {
            margin-top: 10px;
        }
        
        .test-traceback summary {
            cursor: pointer;
            color: #667eea;
            font-weight: 500;
        }
        
        .test-traceback pre {
            margin-top: 10px;
            padding: 15px;
            background: #263238;
            color: #aed581;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.85rem;
        }
        
        .performance-section h2 {
            font-size: 1.8rem;
            margin-bottom: 20px;
            color: #667eea;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #e0e0e0;
        }
        
        .metric-card.success {
            border-color: #4caf50;
        }
        
        .metric-card.danger {
            border-color: #f44336;
        }
        
        .metric-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .metric-icon {
            font-size: 1.5rem;
        }
        
        .metric-name {
            font-weight: 600;
            font-size: 1.1rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .metric-threshold {
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 10px;
        }
        
        .metric-description {
            font-size: 0.9rem;
            color: #666;
        }
        
        .issues-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .issues-card, .recommendations-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
        }
        
        .issues-card h2, .recommendations-card h2 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #667eea;
        }
        
        .issues-list, .recommendations-list {
            list-style: none;
        }
        
        .issue-item, .recommendation-item {
            padding: 12px;
            margin: 8px 0;
            background: white;
            border-radius: 6px;
            border-left: 4px solid #2196f3;
        }
        
        .issue-item.danger {
            border-left-color: #f44336;
        }
        
        .status-icon {
            font-size: 1.3rem;
            margin-right: 10px;
        }
        
        @media (max-width: 768px) {
            .summary-section {
                grid-template-columns: 1fr;
            }
            
            .summary-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .issues-section {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{TITLE}}</h1>
            <div class="timestamp">Generated on {{TIMESTAMP}}</div>
        </div>
        
        <div class="content">
            {{SUMMARY}}
            {{TEST_SUITES}}
            {{PERFORMANCE}}
            {{ISSUES}}
        </div>
    </div>
</body>
</html>
        """
