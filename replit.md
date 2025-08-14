# SafeData 2.0 - Advanced Data Anonymization Platform

## Overview

SafeData 2.0 is a comprehensive data privacy and anonymization platform that implements federated learning-based privacy-preserving techniques. The system provides advanced data anonymization capabilities using synthetic data generation (SDG), differential privacy (DP), and statistical disclosure control (SDC). It features a FastAPI backend with a web-based frontend for data upload, anonymization, and privacy assessment.

The platform serves as a government-grade solution for organizations needing to anonymize sensitive datasets while maintaining data utility for analysis and machine learning applications.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: FastAPI with async/await support for high-performance API endpoints
- **Modular Design**: Service-oriented architecture with separate services for anonymization, privacy assessment, audit logging, and attack simulation
- **Router Pattern**: API endpoints organized by functionality (anonymize, privacy, audit) with clear separation of concerns

### Privacy Engine Components
- **Anonymization Engine**: Core service orchestrating multiple privacy techniques including synthetic data generation, differential privacy, and statistical disclosure control
- **Differential Privacy Engine**: Implements Gaussian and Laplace mechanisms with privacy budget tracking using TensorFlow Privacy
- **Synthetic Data Generator**: Uses advanced deep learning models (CTGAN, TVAE, CopulaGAN) via SDV library for generating privacy-preserving synthetic datasets
- **Attack Simulator**: Evaluates anonymization effectiveness through simulated privacy attacks (linkage, membership, attribute inference)

### Security and Compliance
- **Security Manager**: Handles encryption, secure hashing, and session token generation using cryptography library
- **Audit Service**: Comprehensive logging system for compliance with detailed event tracking, privacy budget monitoring, and session management
- **Data Validation**: Multi-layer validation including file type checking, statistical validation, and sensitive data detection

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Bootstrap dark theme for responsive UI
- **Static Assets**: CSS and JavaScript files for interactive dashboard, file upload, and result visualization
- **Chart Integration**: Chart.js for privacy metrics and utility visualization

### File Processing Pipeline
- **File Handler**: Secure file processing supporting CSV, Excel, and PDF formats with magic number validation
- **Memory Storage**: In-memory data storage for demo purposes (designed for database integration in production)
- **Background Processing**: Async background tasks for long-running anonymization operations

### Configuration Management
- **Settings System**: Centralized configuration using Pydantic BaseSettings with environment variable support
- **Privacy Parameters**: Configurable epsilon/delta values, synthetic data training parameters, and attack simulation settings
- **Security Settings**: Encryption keys, session management, and audit retention policies

## External Dependencies

### Core Libraries
- **FastAPI**: Modern Python web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning utilities for attack simulation and validation

### Privacy Libraries
- **TensorFlow Privacy**: Google's differential privacy library for implementing DP mechanisms
- **SDV (Synthetic Data Vault)**: Advanced synthetic data generation using deep learning models
- **Scikit-optimize**: Bayesian optimization for privacy-utility trade-off tuning

### Security and Validation
- **Cryptography**: Advanced encryption and cryptographic operations
- **Pydantic**: Data validation and settings management
- **python-magic**: File type detection and validation

### Frontend Dependencies
- **Bootstrap**: Responsive CSS framework with dark theme support
- **Chart.js**: Interactive charting library for data visualization
- **Feather Icons**: Lightweight icon library for UI elements

### Optional Dependencies
- **Camelot**: PDF table extraction capabilities
- **Python-multipart**: File upload handling
- **Jinja2**: Template engine for HTML rendering

### Development and Deployment
- **Logging**: Built-in Python logging with configurable levels
- **CORS Middleware**: Cross-origin resource sharing support
- **Static File Serving**: FastAPI's built-in static file handling