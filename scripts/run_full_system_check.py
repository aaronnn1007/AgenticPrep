"""
Master Integration Test Runner
===============================
Production-grade test orchestrator for comprehensive system validation.

Features:
- Runs all integration test suites
- Collects and aggregates results
- Generates HTML report
- Performance profiling
- Error analysis
- Actionable recommendations

Usage:
    python scripts/run_full_system_check.py
    python scripts/run_full_system_check.py --fast  # Quick smoke tests
    python scripts/run_full_system_check.py --performance-only  # Performance tests only
"""

from utils.html_report_generator import (
    HTMLReportGenerator,
    IntegrationReport,
    TestSuite,
    TestResult,
    TestStatus,
    PerformanceMetric,
    SystemInfo
)
import argparse
import asyncio
import logging
import platform
import sys
import time
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integration_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """
    Master integration test runner.

    Orchestrates all integration tests and generates comprehensive report.
    """

    def __init__(self, fast_mode: bool = False, performance_only: bool = False):
        """
        Initialize test runner.

        Args:
            fast_mode: Run quick smoke tests only
            performance_only: Run performance tests only
        """
        self.fast_mode = fast_mode
        self.performance_only = performance_only
        self.start_time = time.time()

        # Test suites to run
        self.test_suites = []
        self.performance_metrics = []
        self.issues = []
        self.recommendations = []

    def run_all_tests(self) -> IntegrationReport:
        """
        Run all integration tests.

        Returns:
            Complete integration report
        """
        logger.info("=" * 80)
        logger.info("Starting Full System Integration Tests")
        logger.info("=" * 80)

        # Check prerequisites
        self._check_prerequisites()

        # Run test suites
        if self.performance_only:
            self._run_performance_tests()
        elif self.fast_mode:
            self._run_smoke_tests()
        else:
            self._run_all_test_suites()

        # Generate report
        report = self._generate_report()

        # Generate HTML
        html_path = project_root / "integration_report.html"
        generator = HTMLReportGenerator()
        generator.generate_report(report, html_path)

        logger.info("=" * 80)
        logger.info(f"Integration Tests Complete")
        logger.info(
            f"Total Duration: {report.system_info.total_duration:.2f}s")
        logger.info(f"Success Rate: {report.overall_success_rate:.1f}%")
        logger.info(f"Report: {html_path}")
        logger.info("=" * 80)

        return report

    def _check_prerequisites(self):
        """Check system prerequisites."""
        logger.info("Checking prerequisites...")

        # Check Python version
        python_version = sys.version.split()[0]
        logger.info(f"Python version: {python_version}")

        # Check Redis connection
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379,
                            db=1, socket_timeout=2)
            r.ping()
            logger.info("✓ Redis connection OK")
        except Exception as e:
            logger.error(f"✗ Redis connection failed: {e}")
            self.issues.append(f"Redis connection failed: {e}")
            self.recommendations.append(
                "Install and start Redis server. See WINDOWS_REDIS_INSTALL.md"
            )

        # Check required packages
        required_packages = [
            "fastapi",
            "pytest",
            "pytest-asyncio",
            "faster-whisper",
            "librosa",
            "numpy",
            "pydantic",
            "langgraph"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.error(f"✗ Missing packages: {missing_packages}")
            self.issues.append(
                f"Missing packages: {', '.join(missing_packages)}")
            self.recommendations.append(
                f"Install missing packages: pip install {' '.join(missing_packages)}"
            )
        else:
            logger.info("✓ All required packages installed")

    def _run_all_test_suites(self):
        """Run all test suites."""
        logger.info("\nRunning all test suites...")

        # 1. Live Stream Integration Tests
        self._run_test_suite(
            name="Live Stream Integration",
            module="tests.integration.live_stream_test",
            description="Tests for live streaming, transcription, and Redis synchronization"
        )

        # 2. LangGraph Orchestration Tests
        self._run_test_suite(
            name="LangGraph Orchestration",
            module="tests.integration.orchestration_test",
            description="Tests for pipeline execution order and state propagation"
        )

        # 3. Frontend-Backend Contract Tests
        self._run_test_suite(
            name="Frontend-Backend Contract",
            module="tests.integration.frontend_contract_test",
            description="Tests for API schema validation and field consistency"
        )

        # 4. Edge Cases, Consistency, and Performance Tests
        self._run_test_suite(
            name="Edge Cases & Consistency",
            module="tests.integration.edge_consistency_performance_test",
            description="Tests for edge cases, deterministic behavior, and performance"
        )

    def _run_smoke_tests(self):
        """Run quick smoke tests."""
        logger.info("\nRunning smoke tests (fast mode)...")

        # Run subset of critical tests
        self._run_test_suite(
            name="Smoke Tests",
            module="tests.integration.live_stream_test",
            description="Quick validation of critical functionality",
            test_filter="test_redis_session_creation_and_retrieval or test_voice_metrics_computation"
        )

    def _run_performance_tests(self):
        """Run performance tests only."""
        logger.info("\nRunning performance tests...")

        self._run_test_suite(
            name="Performance Tests",
            module="tests.integration.edge_consistency_performance_test",
            description="Performance benchmarking and latency tests",
            test_filter="test_performance"
        )

    def _run_test_suite(
        self,
        name: str,
        module: str,
        description: str,
        test_filter: Optional[str] = None
    ):
        """
        Run a test suite using pytest.

        Args:
            name: Test suite name
            module: Python module path
            description: Test suite description
            test_filter: Optional pytest filter (e.g., "-k test_name")
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*80}")

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            module.replace(".", "/") + ".py",
            "-v",
            "--tb=short",
            "--color=yes",
            "-s"
        ]

        if test_filter:
            cmd.extend(["-k", test_filter])

        # Run pytest
        suite_start = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            suite_duration = time.time() - suite_start

            # Parse results
            suite = self._parse_pytest_output(
                name,
                description,
                result.stdout,
                suite_duration
            )

            self.test_suites.append(suite)

            # Log summary
            logger.info(f"\n{name} Summary:")
            logger.info(f"  Passed: {suite.passed_count}")
            logger.info(f"  Failed: {suite.failed_count}")
            logger.info(f"  Duration: {suite.total_duration:.2f}s")

            # Collect issues
            if suite.failed_count > 0:
                self.issues.append(
                    f"{name}: {suite.failed_count} test(s) failed"
                )
                self.recommendations.append(
                    f"Review failures in {name} test suite and check logs"
                )

        except subprocess.TimeoutExpired:
            logger.error(f"{name} timed out after 600 seconds")

            # Create failed suite
            suite = TestSuite(
                name=name,
                description=description,
                tests=[TestResult(
                    name="Timeout",
                    status=TestStatus.ERROR,
                    duration=600.0,
                    error="Test suite exceeded 10 minute timeout"
                )],
                total_duration=600.0
            )

            self.test_suites.append(suite)
            self.issues.append(f"{name}: Test suite timeout")
            self.recommendations.append(
                f"Investigate performance issues in {name}"
            )

        except Exception as e:
            logger.error(f"Error running {name}: {e}")

            suite = TestSuite(
                name=name,
                description=description,
                tests=[TestResult(
                    name="Error",
                    status=TestStatus.ERROR,
                    duration=0.0,
                    error=str(e)
                )],
                total_duration=0.0
            )

            self.test_suites.append(suite)
            self.issues.append(f"{name}: {str(e)}")

    def _parse_pytest_output(
        self,
        name: str,
        description: str,
        output: str,
        duration: float
    ) -> TestSuite:
        """
        Parse pytest output to extract test results.

        Args:
            name: Test suite name
            description: Test suite description
            output: Pytest stdout
            duration: Total duration

        Returns:
            TestSuite with parsed results
        """
        suite = TestSuite(
            name=name,
            description=description,
            total_duration=duration
        )

        # Parse test results from output
        # This is a simple parser - in production, use pytest-json-report plugin

        lines = output.split('\n')

        for line in lines:
            # Look for test results
            if '::test_' in line and ('PASSED' in line or 'FAILED' in line or 'ERROR' in line):
                parts = line.split('::')
                if len(parts) >= 2:
                    test_name = parts[-1].split()[0]

                    if 'PASSED' in line:
                        status = TestStatus.PASSED
                    elif 'FAILED' in line:
                        status = TestStatus.FAILED
                    else:
                        status = TestStatus.ERROR

                    # Extract duration if present
                    test_duration = 0.0
                    if '[' in line and 's]' in line:
                        try:
                            duration_str = line.split('[')[1].split('s]')[0]
                            test_duration = float(duration_str)
                        except:
                            pass

                    test = TestResult(
                        name=test_name,
                        status=status,
                        duration=test_duration
                    )

                    suite.tests.append(test)

        # If no tests parsed, add summary
        if not suite.tests:
            # Check for summary line
            for line in lines:
                if 'passed' in line.lower() or 'failed' in line.lower():
                    suite.tests.append(TestResult(
                        name="Summary",
                        status=TestStatus.PASSED if 'failed' not in line.lower() else TestStatus.FAILED,
                        duration=duration,
                        message=line.strip()
                    ))
                    break

        return suite

    def _generate_report(self) -> IntegrationReport:
        """Generate integration report."""
        total_duration = time.time() - self.start_time

        # Collect system info
        system_info = SystemInfo(
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            hostname=socket.gethostname(),
            test_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_duration=total_duration
        )

        # Add performance metrics
        self._collect_performance_metrics()

        # Add general recommendations
        self._add_general_recommendations()

        # Create report
        report = IntegrationReport(
            title="Multi-Agent Live Interview Performance Analyzer - Integration Test Report",
            system_info=system_info,
            test_suites=self.test_suites,
            performance_metrics=self.performance_metrics,
            issues=self.issues,
            recommendations=self.recommendations
        )

        return report

    def _collect_performance_metrics(self):
        """Collect performance metrics from test results."""
        # Analyze test durations
        all_durations = []
        for suite in self.test_suites:
            all_durations.extend([t.duration for t in suite.tests])

        if all_durations:
            avg_test_duration = sum(all_durations) / len(all_durations)

            self.performance_metrics.append(PerformanceMetric(
                name="Average Test Duration",
                value=avg_test_duration,
                unit="seconds",
                threshold=5.0,
                passed=avg_test_duration < 5.0,
                description="Average duration per individual test"
            ))

        # Check for performance test suite
        for suite in self.test_suites:
            if "Performance" in suite.name or "Edge Cases" in suite.name:
                self.performance_metrics.append(PerformanceMetric(
                    name=f"{suite.name} Duration",
                    value=suite.total_duration,
                    unit="seconds",
                    threshold=120.0,
                    passed=suite.total_duration < 120.0,
                    description=f"Total duration for {suite.name} suite"
                ))

    def _add_general_recommendations(self):
        """Add general recommendations based on results."""
        total_tests = sum(suite.total_count for suite in self.test_suites)
        total_failed = sum(suite.failed_count for suite in self.test_suites)

        if total_failed == 0:
            self.recommendations.append(
                "✓ All tests passed! System is ready for production."
            )
        elif total_failed <= total_tests * 0.1:
            self.recommendations.append(
                "⚠ Minor issues detected. Review failed tests before deployment."
            )
        else:
            self.recommendations.append(
                "✗ Significant issues detected. System requires attention before deployment."
            )

        # Performance recommendations
        total_duration = time.time() - self.start_time
        if total_duration > 600:  # 10 minutes
            self.recommendations.append(
                "Consider optimizing slow tests or running them in parallel."
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive integration tests for Multi-Agent Interview Analyzer"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run quick smoke tests only"
    )
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Run performance tests only"
    )

    args = parser.parse_args()

    # Run tests
    runner = IntegrationTestRunner(
        fast_mode=args.fast,
        performance_only=args.performance_only
    )

    try:
        report = runner.run_all_tests()

        # Exit with appropriate code
        if report.total_failed > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
