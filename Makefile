# Makefile for Fractal Dimension Analysis Tool
# Provides convenient commands for installation, testing, and validation

.PHONY: help install install-dev install-minimal test test-quick validate clean examples all

# Default target
help:
	@echo "Fractal Dimension Analysis Tool - Make Commands"
	@echo "=============================================="
	@echo ""
	@echo "Installation Commands:"
	@echo "  install          Install full version with all dependencies"
	@echo "  install-minimal  Install minimal version (core functionality only)"
	@echo "  install-dev      Install development version with testing tools"
	@echo "  install-conda    Install using conda environment"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test            Run comprehensive compatibility tests"
	@echo "  test-quick      Run quick validation (Koch curve test)"
	@echo "  test-examples   Test all example scripts"
	@echo "  validate        Validate installation with all fractals"
	@echo ""
	@echo "Example Commands:"
	@echo "  examples        Run all example analyses"
	@echo "  koch-test       Test Koch curve generation and analysis"
	@echo "  rt-demo         Demonstrate RT interface analysis"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean           Clean generated files and cache"
	@echo "  requirements    Update requirements files"
	@echo "  all             Install + test + validate (complete setup)"

# Installation targets
install:
	@echo "📦 Installing full version..."
	pip install -r requirements.txt
	@echo "✅ Full installation complete"

install-minimal:
	@echo "📦 Installing minimal version..."
	pip install -r requirements_minimal.txt
	@echo "✅ Minimal installation complete"

install-dev:
	@echo "📦 Installing development version..."
	pip install -r requirements.txt
	pip install pytest jupyter black flake8
	@echo "✅ Development installation complete"

install-conda:
	@echo "📦 Installing with conda..."
	conda env create -f environment.yml
	@echo "✅ Conda environment created"
	@echo "   Activate with: conda activate fractal-analysis"

# Testing targets
test:
	@echo "🔬 Running comprehensive compatibility tests..."
	python test_python_versions.py

test-quick:
	@echo "⚡ Running quick validation..."
	python fractal_analyzer.py --generate koch --level 5
	@echo "✅ Quick test complete"

test-examples:
	@echo "🧪 Testing all example scripts..."
	python test_examples.py

validate:
	@echo "🔍 Validating installation with multiple fractals..."
	@echo "Testing Koch curve..."
	python fractal_analyzer.py --generate koch --level 5 --analyze_linear_region
	@echo "Testing Sierpinski triangle..."
	python fractal_analyzer.py --generate sierpinski --level 4 --analyze_linear_region
	@echo "Testing Hilbert curve..."
	python fractal_analyzer.py --generate hilbert --level 4 --analyze_linear_region
	@echo "✅ Validation complete"

# Example targets
examples:
	@echo "🧮 Running example analyses..."
	$(MAKE) koch-test
	$(MAKE) hilbert-demo
	@echo "✅ All examples complete"

koch-test:
	@echo "🔄 Testing Koch curve analysis..."
	python fractal_analyzer.py --generate koch --level 5 --analyze_linear_region --eps_plots
	@echo "   Expected dimension: ~1.2619"

hilbert-demo:
	@echo "📐 Demonstrating Hilbert curve analysis..."
	python fractal_analyzer.py --generate hilbert --level 4 --analyze_linear_region
	@echo "   Expected dimension: ~2.000"

rt-demo:
	@echo "🌊 RT interface analysis demo..."
	@echo "   (Note: Requires RT interface data file)"
	@echo "   Example command: python fractal_analyzer.py --file RT160x200_interface.txt --rt_interface"

# Utility targets
clean:
	@echo "🧹 Cleaning generated files..."
	rm -f *.txt *.png *.eps
	rm -rf __pycache__ .pytest_cache
	rm -rf .numba_cache
	find . -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

requirements:
	@echo "📋 Updating requirements files..."
	pip freeze > requirements_frozen.txt
	@echo "✅ Requirements updated"

# Development targets
format:
	@echo "🎨 Formatting code..."
	black fractal_analyzer.py test_python_versions.py
	@echo "✅ Code formatted"

lint:
	@echo "🔍 Linting code..."
	flake8 fractal_analyzer.py --max-line-length=100 --ignore=E203,W503
	@echo "✅ Linting complete"

# Complete setup
all: install test validate
	@echo "🎉 Complete setup finished!"
	@echo ""
	@echo "Ready for fractal analysis! Try:"
	@echo "  python fractal_analyzer.py --generate koch --level 5"

# Publication support targets
pub-test:
	@echo "📄 Testing publication reproducibility..."
	python fractal_analyzer.py --generate koch --level 5 --eps_plots --no_titles
	python fractal_analyzer.py --generate sierpinski --level 4 --eps_plots --no_titles
	@echo "✅ Publication figures generated"

# Docker targets (if Docker is available)
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t fractal-analyzer .
	@echo "✅ Docker image built"

docker-test:
	@echo "🐳 Testing Docker container..."
	docker run --rm fractal-analyzer python fractal_analyzer.py --generate koch --level 3
	@echo "✅ Docker test complete"

# Help for specific use cases
help-reviewer:
	@echo "📚 Instructions for Publication Reviewers"
	@echo "========================================"
	@echo ""
	@echo "Quick validation of publication results:"
	@echo "  make install-minimal"
	@echo "  make test-quick"
	@echo ""
	@echo "Full validation:"
	@echo "  make install"
	@echo "  make validate"

help-researcher:
	@echo "🔬 Instructions for Researchers"
	@echo "==============================="
	@echo ""
	@echo "Full installation with all features:"
	@echo "  make install"
	@echo "  make validate"
	@echo ""
	@echo "Development setup:"
	@echo "  make install-dev"
	@echo "  make test"

help-conda:
	@echo "🐍 Conda Installation Instructions"
	@echo "=================================="
	@echo ""
	@echo "Create conda environment:"
	@echo "  make install-conda"
	@echo "  conda activate fractal-analysis"
	@echo ""
	@echo "Or manually:"
	@echo "  conda env create -f environment.yml"
	@echo "  conda activate fractal-analysis"
